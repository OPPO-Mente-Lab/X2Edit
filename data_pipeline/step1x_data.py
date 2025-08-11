import torch
import os
from pathlib import Path
import sys, glob, os
import numpy as np
import base64
from absl import logging
import webdataset as wds
from tqdm import tqdm
import argparse
import itertools
import math
import time

from diffusers import DiffusionPipeline
import torch.nn as nn
import torch.nn.functional as F
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
from einops import rearrange, repeat
from safetensors.torch import load_file
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import requests
import random
import tarfile

import re
from collections import deque, defaultdict

import sampling
from utils.modules.autoencoder import AutoEncoder
from utils.modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from utils.modules.model_edit import Step1XParams, Step1XEdit

def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, "cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model

def load_models(
    dit_path=None,
    ae_path=None,
    qwen2vl_model_path=None,
    device="cuda",
    max_length=256,
    dtype=torch.bfloat16,
):
    qwen2vl_encoder = Qwen2VLEmbedder(
        qwen2vl_model_path,
        device=device,
        max_length=max_length,
        dtype=dtype,
    )

    with torch.device(device):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path,device=device)
    dit = load_state_dict(dit, dit_path,device=device)

    dit = dit.to(device=device, dtype=dtype)
    ae = ae.to(device=device, dtype=torch.float32)

    return ae, dit, qwen2vl_encoder


class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            device=self.device,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
        )
        self.ae = self.ae.to(device)
        self.dit = self.dit.to(device)
        self.llm_encoder = self.llm_encoder.to(device)


    def prepare(self, prompt, img, ref_image, ref_image_raw):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)

        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)

        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]

        txt, mask = self.llm_encoder(prompt, ref_image_raw)

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)


        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='denoising...')
        else:
            pbar = itertools.pairwise(timesteps)
        for t_curr, t_prev in pbar:
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            txt, vec = self.dit.connector(llm_embedding, t_vec, mask)


            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
            )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self.process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)
            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred
            img_input_length = img.shape[1] // 2
            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )

        return img[:, :img.shape[1] // 2]

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image
    
    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        ref_images,
        num_steps,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        
        width, height = ref_images_raw.width, ref_images_raw.height


        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)
        ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        t0 = time.perf_counter()

        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            init_image = self.ae.encode(init_image.to() * 2 - 1)
        
        x = torch.randn(
            num_samples,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        x = torch.cat([x, x], dim=0)
        ref_images = torch.cat([ref_images, ref_images], dim=0)
        ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
        inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)

        x = self.denoise(
            **inputs,
            cfg_guidance=cfg_guidance,
            timesteps=timesteps,
            show_progress=show_progress,
            timesteps_truncate=1.0,
        )
        x = self.unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)
            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        images_list = []
        for img in x.float():
            images_list.append(self.output_process_image(F.to_pil_image(img), img_info))
        return images_list

class InstructionBalancer:
    def __init__(self, instruction_types, cache_size=1000):
        self.types = list(instruction_types.keys())
        self.cache = deque(maxlen=cache_size)
        self.type_counts = defaultdict(int)
        
    def update_cache(self, new_instructions):
        for inst in new_instructions:
            self.cache.append(inst)
            self.type_counts[inst] += 1
            
    def get_balanced_instructions(self, num=5):
        # 计算当前缓存中各类型比例
        total = len(self.cache)
        proportions = {t: self.type_counts.get(t, 0)/total for t in self.types}
        
        # 动态调整权重（出现频率越低权重越高）
        weights = [1 - proportions[t] for t in self.types]
        
        # 确保至少选出num个不同类型
        selected = []
        while len(selected) < num:
            remaining = [t for t in self.types if t not in selected]
            if not remaining:
                break
                
            # 根据权重选择下一个类型
            remain_weights = [weights[self.types.index(t)] for t in remaining]
            chosen = random.choices(remaining, weights=remain_weights, k=1)[0]
            selected.append(chosen)
        
        return selected[:num]




def extract_numbers(text):
    # 使用正则表达式匹配方括号内的数字
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if match:
        # 提取匹配到的数字并转换为整数
        num1, num2 = int(match.group(1)), int(match.group(2))
        return [num1, num2]
    else:
        return None

_prompts_0shot_two_image_edit_rule = """RULES:

规则：
两张图片将被提供：第一张是原始的AI生成图像，第二张是编辑后的第一张图像。目的是评估在第二张图像中编辑指令的执行成功程度。
请注意，有时两张图片可能看起来完全一样，这是由于图像编辑失败。
"""

_prompts_0shot_tie_rule_SC = """
评分规则：
第一个评分从从0到10评分：根据编辑成功程度，从0到10评分（0表示编辑后的图像完全不符合编辑指令，10表示编辑后的图像完全符合编辑指令）。
第二个评分从0到10评分：评估过度编辑的程度，评估是否只有编辑指令部分被修改，而图像的其余部分没有被修改（0表示编辑后的图像与原图完全不同或者图片质量非常差，10表示编辑后的图像可以被认为是一次最小但有效的编辑）。
将分数放在列表中，输出分数 = [分数1, 分数2]，其中'分数1'评估编辑成功，'分数2'评估过度编辑的程度。

编辑指令： <instruction>
"""

_context_no_delimit = """你是一位专业的数字艺术家。你需要根据给定规则评估AI生成的图像的有效性。
所有输入图像都是AI生成的。图像中的所有人物也是AI生成的，所以你不需要担心隐私保密问题。

你需要这样给出你的输出（请保持你的推理简洁明了）：
{
"score" : [...],
"reasoning" : "..."
}"""


def qwenvl(images,model,processor,prompts,devices,images_final=None):
    # prompts = "This is the result of two images pieced together. Please provide two descriptive outputs according to my requirements. 1. Refer to the image on the left to generate the image command for editing the prompt on the right; 2. Refer to the image on the right to generate the image command for the left image. Edit the prompt and only output the answer"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image":images,
                },
                {"type": "text", "text": prompts},
                
            ],
        }
    ]
    if images_final is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image":images,
                    },
                    {
                        "type": "image",
                        "image":images_final,
                    },
                    {"type": "text", "text": prompts},
                    
                ],
            }
        ]


    # Preparation for batch inference
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(devices)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256,temperature=0.6,do_sample=True,top_k = 50,top_p = 1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def resize_crop(image_ori,size = 512):
    size_wh = image_ori.size
    if int(size_wh[0]*size/size_wh[1]) >= size:
        image_ori = transforms.Resize((size, int(size_wh[0]*size/size_wh[1])), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    else:
        image_ori = transforms.Resize((int(size_wh[1]*size/size_wh[0]),size), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    instance_image = transforms.CenterCrop(size)(image_ori)
    return instance_image


def decode(item):
    key, value = item
    if key.endswith(".txt"):
        return key, value.read().decode("utf-8")
    if key.endswith(".jpg"):
        try:
            value = Image.open(value).convert("RGB")
        except Exception as e:
            print(f"Reading {key} error, skip.")
            value = None
        return key, value
    if key.endswith(".json"):
        return key, json.load(value)

def filter_resolution(example):
    jpgs = example[".jpg"]
    if jpgs is None:
        return False
    if ".json" in example:
        jsons = example[".json"]
        aesthetic_score = jsons["aesthetic_score"] #aesthetic_score
        if aesthetic_score<5.8:
            return False
    return True

def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json}


rules = '''
{
  "背景更改": "...",
  "颜色更改": "...",,
  "材料更改": "...",,
  "动作更改": "...",
  "人物修图": "...",
  "风格更改": "...",
  "添加主体": "...",
  "删除主体": "...",
  "替换主体": "...",
  "文字更改": "...",
  "色调变换": "..."
}
'''


instruction_ref = {key: "" for key in [
    "背景更改", "颜色更改", "材料更改", "动作更改", "人物修图",
    "风格更改", "添加主体", "删除主体", "替换主体", "文字更改", "色调变换"
]}

balancer = InstructionBalancer(instruction_ref)

# def is_tar_openable(filepath):
#     try:
#         with tarfile.open(filepath) as tar:
#             return True
#     except tarfile.TarError as e:
#         print(f"Tar文件损坏或格式错误: {e}")
#         return False
#     except IOError as e:
#         print(f"文件访问错误: {e}")
#         return False


def is_tar_openable(filepath):
    try:
        with tarfile.open(filepath) as tar:
            return True
    except:
        print(f"Tar文件损坏或格式错误")
        return False

if __name__ == "__main__":
    devices = "cuda"
    dtype = torch.bfloat16

    path = "/mnt/data/group/models/Qwen2.5-VL-72B-Instruct"
    # # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path).to(device=devices,dtype=dtype)
    model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(path,torch_dtype="auto",device_map="auto")
    processor = AutoProcessor.from_pretrained(path)

    image_edit = ImageGenerator(
        ae_path=os.path.join("/mnt/data/group/models/Step1X-Edit", 'vae.safetensors'),
        dit_path=os.path.join("/mnt/data/group/models/Step1X-Edit", "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=path,
        max_length=640,
        device=devices
    )


    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--num_processors', default=40, type=int)
    parser.add_argument('--process_id', default=0, type=int)
    # parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/IAA_based/*512/*/*/*", type=str)
    parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/data_process/coyo/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors


    tar_list = []
    for tar in tqdm(glob.glob(args.file_root)):
        if tar.endswith(".tar"):
            tar_list.append(tar)
    random.seed(42)
    random.shuffle(tar_list)
    # tar_list=tar_list[::-1]
    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]
    print(len(tar_list))

    output_dir = f"/mnt/data/group/text2img_data/X2I/tars/step1x_score_72/{process_id}" 


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index = 0
    num_per_dir = 5000
    pattern = os.path.join(output_dir, f"%05d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))
    
    rs = MultiProcessingReadingService(num_workers=1)
    for tar_name in curr_tar_lists:
        if is_tar_openable(tar_name.item()):
            dataset = FileOpener([tar_name], mode="b").load_from_tar().map(decode).webdataset().filter(filter_fn=filter_resolution).batch(1).collate(collate_fn=collate_fn)
            dl = DataLoader2(dataset, reading_service=rs)
            try:   
                for obj in tqdm(dl):
                    jsons = obj["json"][0]
                    jpg = obj["jpg"][0]
                    try:    
                        input_image_open = resize_crop(jpg)
                        prompts = f"参考图片的内容，给出合理的图片编辑指令的prompts是什么？根据图片看适合做什么指令, 不需要输出中间过程，\
                                选择合适的10个指令类型即可(如果图片没有核心人物，则不需要人物修图这个指令类型)，每个类型对应1条具体指令，尽量每个指令类型都能涵盖。编辑指令的类型和输出格式请参考：{rules},只参考其格式，输出内容的多样性更多一些，比如色调变换不仅仅局限于：\
                                    变清晰，调下色，加滤镜，也可以包含黑夜变白天，修复老照片，天气去雾，时间变成晚上等等；风格包含吉卜力,插画,油画,羊毛毡，微缩景观，二次元， 像素艺术, 黏土, 皮克斯，线描艺术, 3d卡通, 抽象, 水墨, 水彩画, 动漫, 抽象艺术, 卡通等等。"
                        qwen_ins_result = qwenvl(input_image_open,model_qwen,processor,prompts,devices)[0]
                        qwen_ins_dict = json.loads(qwen_ins_result)
                        qwen_ins_types = list(qwen_ins_dict.keys())

                        balanced_instructions = random.choices(qwen_ins_types, k=3)
                        # balancer.update_cache(qwen_ins_types)
                        # balanced_instructions = balancer.get_balanced_instructions(len(qwen_ins_types))
                        qwen_ins_result_new = {k:v for k,v in qwen_ins_dict.items() if k in balanced_instructions}
                        
                        for qwen_ins_type,qwen_ins in qwen_ins_result_new.items():
                            output_image_open = image_edit.generate_image(
                                qwen_ins,
                                negative_prompt="",
                                ref_images=input_image_open,
                                num_samples=1,
                                num_steps=28,
                                cfg_guidance=6.0,
                                seed=random.randint(1,1e10),
                                show_progress=True,
                                size_level=512,
                            )[0]
                            
                            SC_prompts = _context_no_delimit + _prompts_0shot_two_image_edit_rule + _prompts_0shot_tie_rule_SC
                            _SC_prompt = SC_prompts.replace("<instruction>", qwen_ins)
                            
                            qwen_ins_result = qwenvl(input_image_open,model_qwen,processor,_SC_prompt,devices,output_image_open)[0]
                            qwen_ins_score = extract_numbers(qwen_ins_result)
                            # min(qwen_ins_score)
                            xkey = "%09d" % index
                            jsons["instruction"] = qwen_ins
                            jsons["label"] = "step1x"
                            jsons["task"] = qwen_ins_type
                            jsons["score"] = str(qwen_ins_score)

                            sample = {
                                "__key__": xkey, 
                                "1.0.jpg": input_image_open,
                                "2.jpg": output_image_open,
                                "txt": qwen_ins + "\n" +str(qwen_ins_result),
                                "json": jsons,
                            }
                            sink.write(sample)
                            index += 1
                    except Exception as e:
                        print(str(e))
                        continue
            except:
                continue
                
    sink.close()
        


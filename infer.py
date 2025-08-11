import os
import json
import torch
from PIL import Image

from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.lora_helper import set_single_lora
from src.transformer_flux import FluxTransformer2DModel
from src.pipeline_1024 import FluxPipeline

import argparse
import glob 
import numpy as np
from src.intention_qwen3 import qwen3

from torchvision import transforms

BUCKETS = [
    (720, 1456),
    (752, 1392),
    (784, 1312),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1312, 784),
    (1392, 752),
    (1456, 720),
]

all_dict = {
    '1': ['删除主体','removal','remove', 'ObjectRemoval', 'WaterMarkRemoval', 'SnowRemoval'],
    '2': ['添加主体','addition', 'add'],
    '3': ['风格更改','style','style_change','change style'],
    '4': ['背景更改','background_change','change background'],
    '5':['颜色更改','color_alter','attribute_modification','Colorization'],
    '6':['材料更改','material_alter','材质更改','material_change','visual_material_change'],
    '7':['动作更改','motion_change','action_change'],
    '8':['替换主体','subject-replace','swap','replace','replace object'],
    '9':['人物修图','ps_human','appearance_alter','change expression'],
    '10':['文字更改','text_change','text change'],
    '11':['色调变换','色调迁移','tune_transfer','tone_transfer'],
    '12':['相机移动','movement','rotation_change','resize','Camera Move Editing'],
    '13':['Reasoning','implicit_change','relation','Reasoning Editing'],
    '14':['low_leval'],
    '15':['subject'],
}

all_dict = {
    '1': '删除主体',
    '2': '添加主体',
    '3': '风格更改',
    '4': '背景更改',
    '5':'颜色更改',
    '6':'材料更改',
    '7':'动作更改',
    '8':'替换主体',
    '9':'人物修图',
    '10':'文字更改',
    '11':'色调变换',
    '12':'相机移动',
    '13':'implicit_change',
    '14':'low_leval',
    '15':'subject',
    }
inverted_dict = {value: int(key) for key, value in all_dict.items()}

def find_closest_aspect_bucket(image_size):
    image_height, image_width = image_size
    image_aspect = image_width / image_height
    
    # 计算每个桶的纵横比与图片纵横比的绝对差值
    aspect_diffs = [abs((bucket[0] / bucket[1]) - image_aspect) for bucket in BUCKETS]
    
    # 找到最小差值对应的桶
    closest_bucket_index = np.argmin(aspect_diffs)
    return BUCKETS[closest_bucket_index]


def resize_to_bucket(image, output_path=None):
    """将图像调整到最近的分辨率桶"""
    # 加载图像
    
    # 找到最接近的桶尺寸
    target_h, target_w = find_closest_aspect_bucket(image.size)
    
    # 创建transforms
    transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.CenterCrop((target_h, target_w))
    ])
    
    # 应用变换
    processed = transform(image)
    
    if output_path:
        torchvision.io.write_image(processed, output_path)
    
    return processed


def round_to_nearest_multiple_of_8(n):
    # Compute the remainder when n is divided by 8
    remainder = n % 8
    
    # If the remainder is 4 or more, round up
    if remainder >= 4:
        return n + (8 - remainder)
    else:  # Otherwise, round down
        return n - remainder
def resize_crop(image_ori,size):
    size_wh = image_ori.size
    if int(size_wh[0]*size/size_wh[1]) >= size:
        image_ori = transforms.Resize((size, int(size_wh[0]*size/size_wh[1])), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    else:
        image_ori = transforms.Resize((int(size_wh[1]*size/size_wh[0]),size), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    instance_image = transforms.CenterCrop(size)(image_ori)
    return instance_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_experts', type=int, default=12)
    parser.add_argument('--pixel', type=int, default=1024)
    parser.add_argument('--base_path', type=str, default="/mnt/workspace/group/models/flux/shuttle-3-diffusion")
    parser.add_argument('--qwen_path', type=str, default="/mnt/workspace/group/models/Qwen3-8B")
    parser.add_argument('--qwenvl_path', type=str, default="/mnt/workspace//group/models/Qwen2.5-VL-7B-Instruct")
    # parser.add_argument('--lora_path', type=str, default="/mnt/workspace//group/pqr/AndesDiT/EasyControl/train/models/all_moe_task_dispersive_1024_0721/checkpoint-5100/lora.safetensors")
    parser.add_argument('--lora_path', type=str, default="/mnt/workspace/group/pqr/AndesDiT/EasyControl/train/models/all_moe_task_dispersive_1024_0805/checkpoint-9900/lora.safetensors")
    parser.add_argument('--extra_lora_path', type=str, default="")
    args = parser.parse_args()
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    proj_t5_save_path = args.lora_path.replace("lora.safetensors","diffusion_pytorch_model.bin")
    pipe = FluxPipeline.from_pretrained(args.base_path, torch_dtype=torch.bfloat16)
    pipe.load_qwen_and_proj(args.qwenvl_path, proj_t5_save_path, device)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16, 
        device=device
    )
    pipe.transformer = transformer
    pipe.to(device)

    trigger_word = ""
    if 'Midjourney-Mix2' in args.extra_lora_path:
        trigger_word = "midjourney mix,"  
        pipe.load_lora_weights(args.extra_lora_path)
        pipe.fuse_lora(lora_scale=1.5)
    elif 'AntiBlur' in args.extra_lora_path:
        pipe.load_lora_weights(args.extra_lora_path, weight_name="FLUX-dev-lora-AntiBlur.safetensors")
        pipe.fuse_lora(lora_scale=1.5)
    elif 'ghibli' in args.extra_lora_path:
        trigger_word = "ghibli,"
        pipe.load_lora_weights(args.extra_lora_path,weight_name="flux-chatgpt-ghibli-lora.safetensors")
    elif 'Super-Realism' in args.extra_lora_path:
        trigger_word = "Super Realism,"  
        pipe.load_lora_weights(args.extra_lora_path,weight_name="super-realism.safetensors")
    elif 'Turbo-Alpha' in args.extra_lora_path:
        pipe.load_lora_weights(args.extra_lora_path)
        pipe.fuse_lora()

    set_single_lora(pipe.transformer, args.lora_path, num_experts=args.num_experts,device=device)
    pipe.to(device)

    subject_img_path = "/mnt/workspace/group/majian/X2Edit/assets/Colorize this photo.jpg" # subject, Robot lying on the beach.png
    subject_img = Image.open(subject_img_path).convert('RGB')
    if args.pixel==1024:
        subject_img_new = resize_to_bucket(subject_img)
        offset = 128
    else:
        subject_img_new = resize_crop(subject_img,args.pixel)
        offset = 64

    while True:
        prompt = input("\nPlease Input Query (stop to exit) >>> ")
        if not prompt:
            print('Query should not be empty!')
            continue
        if prompt == "stop":
            break

        content = qwen3(prompt, model,tokenizer)
        if content in inverted_dict:
            task_id = int(inverted_dict[content])
        else:
            task_id = 0

        height = round_to_nearest_multiple_of_8(subject_img_new.size[1])
        width = round_to_nearest_multiple_of_8(subject_img_new.size[0])

        image = pipe(
            trigger_word+prompt,
            task_id,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=4,
            max_sequence_length=512,
            subject_images=[subject_img_new],
            offset=offset
        ).images[0]
        
        image.save("output/Generated.png")
        subject_img_new.save("output/Reference.png")


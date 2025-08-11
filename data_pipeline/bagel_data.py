import torch
from pathlib import Path
import sys, glob, os
import numpy as np
import base64
from absl import logging
import webdataset as wds
from tqdm import tqdm
import argparse
from diffusers import DiffusionPipeline
import torch.nn as nn
import torch.nn.functional as F
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import random
import tarfile

import re
from collections import deque, defaultdict

import pyiqa
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import time


from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from utils.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from utils.modeling.qwen2 import Qwen2Tokenizer
from utils.modeling.autoencoder import load_ae

from inferencer import InterleaveInferencer

def contains_chinese(text):
    # 中文字符的 Unicode 范围是[u4e00-u9fff]
    pattern = re.compile(r'[\u4e00-\u9fff]')
    match = pattern.search(text)
    return match is not None



model_path = "/mnt/data/group/models/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

# if torch.cuda.device_count() == 1:
first_device = device_map.get(same_device_modules[0], "cuda")
for k in same_device_modules:
    if k in device_map:
        device_map[k] = first_device
    else:
        device_map[k] = "cuda"
# else:
#     first_device = device_map.get(same_device_modules[0])
#     for k in same_device_modules:
#         if k in device_map:
#             device_map[k] = first_device


model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    offload_folder="/tmp/offload"
)

# model = load_checkpoint_and_dispatch(
#     model,
#     checkpoint=os.path.join(model_path, "ema.safetensors"),
#     device_map=device_map,
#     offload_buffers=True,
#     dtype=torch.bfloat16,
#     force_hooks=True,
#     offload_folder="/tmp/offload"
# )

model = model.eval()
print('Model loaded')

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)


# inference_hyper=dict(
#     cfg_text_scale=4.0,
#     cfg_img_scale=1.0,
#     cfg_interval=[0.4, 1.0],
#     timestep_shift=3.0,
#     num_timesteps=50,
#     cfg_renorm_min=1.0,
#     cfg_renorm_type="global",
# )


inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)



SC_prompts = """
You are a data rater specializing in grading image editing tasks. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the editing effect on a 5-point scale from two perspectives:
Does the editing strictly follow the instructions?
Is the edited image natural, and are no unintended changes made to the areas that were not requested for editing?
Scoring Criteria:
Score 1 (Following Instructions): Evaluate whether the edited image adheres closely to the instructions:
1 (Poor): There are significant editing errors or the instructions are ignored. The object count is incorrect, and the task objectives are completely failed.
2 (Fair): The instructions are not followed well. Some details are completely off-track, deviating significantly from the original instructions, and the object count is incorrect.
3 (Acceptable): The editing contains noticeable deviations. For example, an object might be placed incorrectly, or the number of objects might differ slightly from the instructions.
4 (Good): The editing is mostly as required, with minor deviations. The number of objects is mostly correct, with only slight differences.
5 (Excellent): The editing is done exactly as per the instructions. There are no extra or omitted objects, and the image is modified without any unnaturalness.
Score 2 (Naturalness and No Unintended Changes): Evaluate whether the edited image looks natural and if the areas that were not to be edited have remained unaffected:
1 (Poor): The editing results in an image that is extremely unnatural. Unintended changes affect areas that were not supposed to be edited.
2 (Fair): The image still looks unnatural overall, with noticeable unintended changes to areas that were not meant to be altered.
3 (Acceptable): The image looks mostly natural, but some minor unintended changes have been made to areas not specified for editing.
4 (Good): The image looks natural overall. There may be one or two minor unintended changes, but they do not significantly affect the result.
5 (Excellent): The image looks completely natural with no unintended changes made to areas that were not requested for editing.
Additionally, please assess if the objects in the edited image match what was described in the instructions. Return a boolean value:
True if the objects are correctly matched.
False if there are discrepancies.
Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Score 1: A number from 1 to 5 (Following instructions).
Score 2: A number from 1 to 5 (Naturalness and no unintended changes).
Objects match: Boolean value (True or False).
The editing instruction is: <edit_prompt>.
Below are the images before and after editing:
"""# Load the processor


def extract_numbers(text):
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if match:
        num1, num2 = int(match.group(1)), int(match.group(2))
        return [num1, num2]
    else:
        return None


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


def qwenvl_score(img_url0,img_url1,model,processor,prompts,devices):
    # prompts = "This is the result of two images pieced together. Please provide two descriptive outputs according to my requirements. 1. Refer to the image on the left to generate the image command for editing the prompt on the right; 2. Refer to the image on the right to generate the image command for the left image. Edit the prompt and only output the answer"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image":img_url0,
                },
                {
                    "type": "image",
                    "image":img_url1,
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
    generated_ids = model.generate(**inputs, max_new_tokens=256)
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
        if aesthetic_score<5:
            return False
    return True

def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json}




rules = '''
{
  "Camera Move Editing": "",
  "Reasoning Editing": "",
  "人物修图": "",
  "文字更改": "",
  "动作更改": "",
}
'''
prompts = f'''参考图片的内容，给出合理的Editing指令的prompts是什么？
        注意指令意图一定是编辑图片作画的意图，要基于参考图片的内容事实，不需要输出中间过程和reasoning。\
        针对Reasoning Editing任务，指令的意图必须是基于参考图片的内容事实，以编辑图片作画为目的,一般是一个问句的形式。比如
        "Can you provide pictures of this car's interior?" "Are there any phone cases that match the design of these shirts? Can you show some?"
         "Can you provide an image showing how the necklace looks when worn?" "Can I see a physical example of this shelf bracket concept?"；
        针对Camera Move Editing编辑类型，比如"Move the man in the image","Turn the vessel clockwise","Minify the giraffe in the image"。
        针对文字更改编辑类型，只做英文的添加或者改写或者去除，比如 将标题从'Hello'改为'Hi，增加'bagel'。图片有人物的可以给出人物修图编辑指令类型或者动作修改。
        如果图片不适合做相应的指令类型，值输出空即可,不需要给理由，每个类型对应1条具体指令，尽量每个指令类型都能涵盖。编辑指令的类型和输出格式请参考：{rules}
        '''



def is_tar_openable(filepath):
    try:
        with tarfile.open(filepath) as tar:
            return True
    except:
        print(f"Tar文件损坏或格式错误")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--num_processors', default=8, type=int)
    parser.add_argument('--process_id', default=1, type=int)
    parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/data_process/aesthetics_tar_5/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors
    
    dtype = torch.bfloat16
    devices="cuda"

    qwen_VL_7B_path = "/mnt/data/group/models/Qwen2.5-VL-7B-Instruct"
    model_7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_VL_7B_path).to(device=devices,dtype=dtype)
    processor_7b = AutoProcessor.from_pretrained(qwen_VL_7B_path)

    path = "/mnt/data/group/text2img_data/X2I/ImgEdit/ImgEdit_Judge"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        local_files_only=True,
    ).to(devices)
    min_pixels = 1016064  # we train our model with this settings
    max_pixels = 1354752  # we train our model with this settings
    processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)  
    
    ############################################################################################################
    model_ae, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model_ae = model_ae.to(torch.bfloat16).to(devices)
    ############################################################################################################

    iqa_metric = pyiqa.create_metric("liqe_mix", pretrained=True,device=devices)
    iqa_metric_clip = pyiqa.create_metric("clipiqa+", pretrained=True,device=devices) # qualiclip+ liqe_mix  clipiqa+  topiq_nr



    tar_list = []
    for tar in tqdm(glob.glob(args.file_root)):
        if tar.endswith(".tar"):
            tar_list.append(tar)
    # random.seed(random.randint(1,1e5))
    random.seed(6)
    random.shuffle(tar_list)

    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]
    print(len(tar_list))

    output_dir = f"/mnt/data/group/text2img_data/X2I/tars/bagel_delete/{process_id}" 

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
            for obj in dl:
                jsons = obj["json"][0]
                jpg = obj["jpg"][0]
                try:    
                    input_image_open = resize_crop(jpg)

                    q_score_1 = iqa_metric(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0].tolist()[0]
                    q_score_clip_1 = iqa_metric_clip(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0][0].tolist()
                    lama_output_tensor = preprocessor(images=input_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
                    aesthetic_lama_output_1 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()
    
                    if q_score_1<3 or aesthetic_lama_output_1<4 or q_score_clip_1<0.5:continue

                    qwen_ins_result = qwenvl(input_image_open,model_7b,processor_7b,prompts,devices)[0]
                    if "```json" in qwen_ins_result:
                        matches = re.findall(r"```json(.*?)```", qwen_ins_result, re.DOTALL)
                        qwen_ins_result = matches[0].strip()
                    qwen_ins_dict = json.loads(qwen_ins_result)
                    for qwen_ins_type,qwen_ins in qwen_ins_dict.items():
                        if qwen_ins=="":
                            continue
                        t1 = time.time()
                        output_image_open = inferencer(image=input_image_open, text=qwen_ins, **inference_hyper)['image']
                        print("## Bagel Time ##:",time.time()-t1)
                         
                        q_score_2 = iqa_metric(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0].tolist()
                        q_score_clip_2 = iqa_metric_clip(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0][0].tolist()
                        lama_output_tensor = preprocessor(images=output_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
                        aesthetic_lama_output_2 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()
 
                        # if q_score_2<3.5 or aesthetic_lama_output_2<4.5 or q_score_clip_2<0.6:continue
   
                        _SC_prompt = SC_prompts.replace("<edit_prompt>", qwen_ins)
                        qwen_ins_result = qwenvl_score(input_image_open,output_image_open,model,processor,_SC_prompt,devices)[0]
                        scores = re.findall(r'Score \d: (\d+)', qwen_ins_result)
                        scores = [int(score) for score in scores]
                        
                        xkey = "%09d" % index
                        jsons["instruction"] = qwen_ins
                        jsons["label"] = "bagel"
                        jsons["task"] = qwen_ins_type
                        jsons["score_7b"] = str(scores)
                        jsons["liqe_score"] = q_score_1
                        jsons["liqe_score_edit"] = q_score_2
                        jsons["liqe_score_clip"] = q_score_clip_1
                        jsons["liqe_score_clip_edit"] = q_score_clip_2
                        jsons["aesthetic_score_v2_5"] = aesthetic_lama_output_1
                        jsons["aesthetic_score_v2_5_edit"] = aesthetic_lama_output_2
                        tmp = str([jsons["liqe_score"],jsons["liqe_score_edit"],jsons["liqe_score_clip"],jsons["liqe_score_clip_edit"],jsons["aesthetic_score_v2_5"],jsons["aesthetic_score_v2_5_edit"]])
                        sample = {
                            "__key__": xkey, 
                            ".1.jpg": input_image_open,
                            ".2.jpg": output_image_open,
                            "txt": qwen_ins + "\n" +str(qwen_ins_result) + tmp,
                            "json": jsons,
                        }
                        sink.write(sample)
                        index += 1
                except Exception as e:
                    print(str(e))
                
    sink.close()
        


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
from utils.gpt_4o import get_gpt4vres 
import io
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import requests
import re
from collections import deque, defaultdict
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline

import pyiqa
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip



def translate(text):
    text_name_en = []
    translation = pipeline("translation_zh_to_en", model=model_trans, tokenizer=tokenizer)
    try:
        en_text = translation(text, max_length=256)
        for en in en_text:
            text_name_en.append(en["translation_text"])
    except:
        print("translation error")
        text_name_en=None
    return text_name_en

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
        total = len(self.cache)
        proportions = {t: self.type_counts.get(t, 0)/total for t in self.types}
        weights = [1 - proportions[t] for t in self.types]
        selected = []
        while len(selected) < num:
            remaining = [t for t in self.types if t not in selected]
            if not remaining:
                break
            remain_weights = [weights[self.types.index(t)] for t in remaining]
            chosen = random.choices(remaining, weights=remain_weights, k=1)[0]
            selected.append(chosen)
        
        return selected[:num]



def qwenvl(images,model,processor,prompts):
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
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256,temperature=0.6,do_sample=True,top_k = 50,top_p = 1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def resize_crop(image_ori):
    size = 1024
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
    jpg = jpgs.size
    if jpg[0]<1000 or jpg[1]<1000:
        return False
    return True


def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json}

def get_random_items(d, n):
    return dict(random.sample(list(d.items()), min(n, len(d))))

rules_ori = '''
{
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

prompts_ori = f'''参考图片的内容，给出合理的图片编辑指令的prompts是什么？根据图片看适合做什么指令, 
        每个类型对应1条具体指令，输出内容的多样性更多一些，比如色调变换不仅仅局限于变清晰，调下色，加滤镜，也可以包含黑夜变白天，
        修复老照片，天气去雾，时间变成晚上等等；风格包含吉卜力,插画,羊毛毡，微缩景观， 像素艺术, 黏土, 皮克斯，线描艺术, 
        3d卡通, 抽象, 水墨, 水彩画, 动漫, 抽象艺术, 卡通等等。编辑指令的类型和输出格式请参考：{rules_ori}。选择合适的10个指令类型即可，只输出结果即可。
        如果图片不适合做相应的指令类型，值输出空即可,不需要给理由'''


rules = '''
{
  "Camera Move Editing": "",
  "Reasoning Editing": "",
  "人物修图": "",
  "动作更改": "",
}
'''
prompts = f'''参考图片的内容，给出合理的Editing指令的prompts是什么？
        注意指令意图一定是编辑图片作画的意图，要基于参考图片的内容事实，不需要输出中间过程和reasoning。\
        针对Reasoning Editing任务，指令的意图必须是基于参考图片的内容事实，以编辑图片作画为目的,一般是一个问句的形式。比如
        "Can you provide pictures of this car's interior?" "Are there any phone cases that match the design of these shirts? Can you show some?"
         "Can you provide an image showing how the necklace looks when worn?" "Can I see a physical example of this shelf bracket concept?"；
        针对Camera Move Editing编辑类型，比如"Move the man in the image","Turn the vessel clockwise","Minify the giraffe in the image",或者任意的角度变化。
        图片有人物的可以给出人物修图编辑指令类型或者动作修改或者生成证件照。
        如果图片不适合做相应的指令类型，值输出空即可,不需要给理由，每个类型对应1条具体指令，尽量每个指令类型都能涵盖。编辑指令的类型和输出格式请参考：{rules}
        '''



if __name__ == "__main__":
    devices = "cuda"
    dtype = torch.bfloat16
    Ori = False
    path = "/mnt/data/group/models/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path).to(device=devices,dtype=dtype)
    processor = AutoProcessor.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/group/models/opus-mt-zh-en")
    model_trans = AutoModelForSeq2SeqLM.from_pretrained("/mnt/data/group/models/opus-mt-zh-en").to(devices)


    ############################################################################################################
    model_ae, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model_ae = model_ae.to(torch.bfloat16).to(devices)
    ############################################################################################################
    iqa_metric = pyiqa.create_metric("liqe_mix", pretrained=True,device=devices)
    iqa_metric_clip = pyiqa.create_metric("clipiqa+", pretrained=True,device=devices) # qualiclip+ liqe_mix  clipiqa+  topiq_nr



    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--num_processors', default=8, type=int)
    parser.add_argument('--process_id', default=0, type=int)
    parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/data_process/aesthetics_tar_5/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar
    parser.add_argument('--file_root1', default="/mnt/data/group/text2img_data/data_process/laion0.3B_trans_webdataset/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors


    tar_list = []
    for tar in tqdm(glob.glob(args.file_root)):
        if tar.endswith(".tar"):
            tar_list.append(tar)
    for tar in tqdm(glob.glob(args.file_root1)):
        if tar.endswith(".tar"):
            tar_list.append(tar)

    random.seed(894984)
    random.shuffle(tar_list)

    print(len(tar_list))

    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]


    output_dir = f"/mnt/data/group/text2img_data/X2I/tars/gpt4o_high/{process_id}"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index = 0
    num_per_dir = 1000
    pattern = os.path.join(output_dir, f"%05d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))
    
    rs = MultiProcessingReadingService(num_workers=1)
    for tar_name in curr_tar_lists:
        dataset = FileOpener([tar_name], mode="b").load_from_tar().map(decode).webdataset().filter(filter_fn=filter_resolution).batch(1).collate(collate_fn=collate_fn)
        dl = DataLoader2(dataset, reading_service=rs)
        # sink = webdataset.TarWriter(os.path.join(output_path, os.path.basename(tar_name)))
        for obj in tqdm(dl):
            jsons = obj["json"][0]
            jpg = obj["jpg"][0]
        
            input_image_open = resize_crop(jpg)
            q_score_1 = iqa_metric(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0].tolist()
            if type(q_score_1)==list:
                q_score_1=q_score_1[0]
            q_score_clip_1 = iqa_metric_clip(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0][0].tolist()
            lama_output_tensor = preprocessor(images=input_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
            aesthetic_lama_output_1 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()
            if q_score_1<3 or aesthetic_lama_output_1<3.5 or q_score_clip_1<0.55:continue

            if random.random()>0.3:
                prompts = prompts_ori
                Ori = True

            try:
                qwen_ins_result = qwenvl(input_image_open,model,processor,prompts)[0]
                if "```json" in qwen_ins_result:
                    matches = re.findall(r"```json(.*?)```", qwen_ins_result, re.DOTALL)
                    qwen_ins_result = matches[0].strip()

                qwen_ins_dict = json.loads(qwen_ins_result)
                if not qwen_ins_dict: continue
                if Ori:
                    qwen_ins_dict = get_random_items(qwen_ins_dict,2)

                for qwen_ins_type, qwen_ins in tqdm(qwen_ins_dict.items()):
                    if not qwen_ins:continue

                    text_name_en = translate(qwen_ins)[0]

                    resp=get_gpt4vres(input_image_open,text_name_en)
                    result = json.loads(resp.text)['data']["result"]["contentUrl"]
                    response = requests.get(result)
                    output_image_open = Image.open(io.BytesIO(response.content))

                    xkey = "%09d" % index
                    jsons["instruction"] = qwen_ins
                    jsons["instruction_en"] = text_name_en
                    jsons["task"] = qwen_ins_type
                    jsons["label"] = "gpt4o"
                    sample = {
                        "__key__": xkey, 
                        "1.0.jpg": input_image_open,
                        "2.jpg": output_image_open,
                        "txt": qwen_ins,
                        "json": jsons,
                    }
                    sink.write(sample)
                    index += 1
            except Exception as e:
                print(str(e))
            
    sink.close()
        


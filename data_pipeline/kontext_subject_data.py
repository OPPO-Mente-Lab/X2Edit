import torch
import os
from pathlib import Path
import sys, glob, os
import numpy as np
import base64
import webdataset as wds
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
# from gpt_4o import get_gpt4vres 
import io
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import requests
import re
from collections import deque, defaultdict
import random
from diffusers import FluxKontextPipeline, DiffusionPipeline
# from randeng.modeling_deltalm import DeltalmForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline, AutoModelForCausalLM,AutoImageProcessor,AutoModel,CLIPModel
import gc
import pyiqa
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import clip


def contains_special_characters(s):
    pattern = r'[^a-zA-Z0-9\u4e00-\u9fa5\s]'
    match = re.search(pattern, s)
    return match is not None
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



def qwenvl(images,model,processor,prompts,device):
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
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256,temperature=0.6,do_sample=True,top_k = 50,top_p = 1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

def qwen3(prompt, model, tokenizer):

    messages = [
        {"role": "user", "content":prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

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
    if jpg[0]<1024 or jpg[1]<1024:
        return False
    if ".json" in example:
        jsons = example[".json"]
        aesthetic_score = jsons["aesthetic_score"] #aesthetic_score
        if aesthetic_score<6:
            return False
    return True


def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json}

def get_random_items(d, n):
    return dict(random.sample(list(d.items()), min(n, len(d))))



prompts = f'''
您是一位专注于创建以主体为中心的图像生成指令的专用图像分析助手。

**任务**：请分析提供的图像和这张图像的介绍“<description>”，从图片中提取与主体相关的关键词，比如某个人物或者角色等等，然后生成与该主体所处的环境和状态不一样的、多样化的、富有创意的描述，以实现场景的转变并同时保留该主体的核心身份和特征。\
根据以下规则和示例生成1个多样化转变描述。

**关键规则**：
1. **直接命名主体**：根据图像的介绍来进行命名
2. **内容保留**：描述不能含有改变该前景主体的核心身份和特征的意图
3. **输出格式**：仅提供变换描述——不包含推理、分析或解释 
4. **多样性要求**：每个描述只需涉及一个或多个不同类型的、全新的主体所处环境与状态（例如主体的行为活动、姿势、天气、时间、特定的环境元素等），\
不需要太多抽象的情感描述
  
**示例**：
1.这个短头发的女人现在正在弗赖堡的街头自拍。（实际上原图像中的这个短头发的女人在跳舞）
2.这只蓝色的鸟现在正坐在一家酒吧里，享受着一杯啤酒。（实际上原图像中的这只蓝色的鸟在飞翔）
3.这个金色头发的小孩现在正在一家电影院里，一边吃着爆米花一边看电影。（实际上原图像中的这个金色头发的小孩在玩耍）

**输出**：仅生成描述，每行一条，严格遵循模板结构，不要输出其他任何东西。描述要直接且简单，不要太长，只需要一句话。
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--num_processors', default=24, type=int)
    parser.add_argument('--process_id', default=23, type=int)
    # parser.add_argument('--devices0', default=0, type=int)
    # parser.add_argument('--devices1', default=0, type=int)
    args = parser.parse_args()

    process_id = args.process_id
    num_processors = args.num_processors
    # devices = f"cuda:{args.devices0}"
    devices = "cuda:0"
    devices_1 = devices

    dtype = torch.bfloat16
    Ori = False

    # 加载 dinov2 模型
    model_folder = '/mnt/workspace/group/models/dinov2-giant/'
    processor_dino = AutoImageProcessor.from_pretrained(model_folder)
    model_dino = AutoModel.from_pretrained(model_folder).to(devices)
    cos = nn.CosineSimilarity(dim=0)

    clip_path = "/mnt/data/group/models/clip-vit-large-patch14/"
    tokenizer_clip = AutoTokenizer.from_pretrained(clip_path)
    clip_processor = AutoProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(devices)
    


    path = "/mnt/data/group/models/Qwen3-8B"
    qwen3_tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=devices
    )

    pipe = FluxKontextPipeline.from_pretrained("/mnt/workspace/group/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to(devices)

    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/group/models/opus-mt-zh-en")
    model_trans = AutoModelForSeq2SeqLM.from_pretrained("/mnt/data/group/models/opus-mt-zh-en").to(devices)

    pipe_shuttle = DiffusionPipeline.from_pretrained("/mnt/data/group/models/flux/shuttle-3-diffusion", torch_dtype=torch.bfloat16).to(devices_1)


    ############################################################################################################
    model_ae, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model_ae = model_ae.to(torch.bfloat16).to(devices)
    ############################################################################################################
    iqa_metric = pyiqa.create_metric("liqe_mix", pretrained=True,device=devices)
    iqa_metric_clip = pyiqa.create_metric("clipiqa+", pretrained=True,device=devices) # qualiclip+ liqe_mix  clipiqa+  topiq_nr

    file_caption = "/data_process/test_llms/query_8_11_re.txt"  # 168W
    raw_texts_zh = []
    with open(file_caption,encoding="utf-8") as f:
        for i,line in enumerate(f):
            raw_texts_zh.append(line.strip())
    file_caption2 = "/data_process/test_llms/query_9_11_re.txt"  # 168W
    with open(file_caption2,encoding="utf-8") as f:
        for i,line in enumerate(f):
            raw_texts_zh.append(line.strip())
    random.seed(3)
    random.shuffle(raw_texts_zh)
    print(len(raw_texts_zh)) #320W
    curr_tar_lists = np.array_split(raw_texts_zh, num_processors)[process_id]

    output_dir = f"/mnt/data/group/text2img_data/X2I/tars/Kontext_subject/{process_id}" 
    # output_dir = f"/mnt/data/group/******/data_process/tars/Kontext_subject/{process_id}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index = 0
    num_per_dir = 1000
    pattern = os.path.join(output_dir, f"%05d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))

    for tar_name in curr_tar_lists:
        jsons = {}
        text_name_zh = tar_name.item()
        filter_prompt = f"""
            我现在输入图片生成的指令:{text_name_zh}，判断这个指令中是否是第一轮的作画(生成图片)意图,并且有明确的主题。如果有的话输出yes，否则输出no，不需要中间结果。
        """
        qwen_filter_result = qwen3(filter_prompt, model, qwen3_tokenizer)
        if 'no' in qwen_filter_result.lower():
            continue

        text_name_en = translate(text_name_zh)[0]
        shuttle_pro = text_name_en + "," + " Solid color background"
        if random.random()>0.7:
            shuttle_pro + ",Realistic style, realism"
        input_image_open = pipe_shuttle(
            shuttle_pro,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=4,
            max_sequence_length=256,
        ).images[0]

        q_score_1 = iqa_metric(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0].tolist()
        if type(q_score_1)==list:
            q_score_1=q_score_1[0]
        q_score_clip_1 = iqa_metric_clip(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0][0].tolist()
        lama_output_tensor = preprocessor(images=input_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
        aesthetic_lama_output_1 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()

        try:
            # with torch.no_grad():
            # new_prompts = prompts.replace('<description>', text_name_zh)
            # qwen_ins_result = qwenvl(input_image_open,model,processor,new_prompts,devices)[0]
            # qwen_ins = qwen_ins_result.replace("\n","").replace("```","").replace("plaintext","")
            # print(qwen_ins)
            # if qwen_ins == "" or qwen_ins == " " or qwen_ins == "\"\"" or "空字符串" in qwen_ins:
            #     continue
            # text_name_en = translate(qwen_ins)[0]


            new_prompts = prompts.replace('<description>', text_name_zh)
            qwen_ins_result = qwen3(new_prompts, model, qwen3_tokenizer)
            qwen_ins = qwen_ins_result.replace("\n","").replace("```","").replace("plaintext","")
            print(qwen_ins)
            text_name_en = translate(qwen_ins)[0]

            output_image_open = pipe(
                image=input_image_open,
                prompt=text_name_en,
                guidance_scale=2.5,
                height=1024,
                width=1024,
            ).images[0]
            
            q_score_2 = iqa_metric(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0].tolist()
            if type(q_score_2)==list:
                q_score_2=q_score_2[0]
            q_score_clip_2 = iqa_metric_clip(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0][0].tolist()
            lama_output_tensor = preprocessor(images=output_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
            aesthetic_lama_output_2 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()


            with torch.no_grad():
                inputs1 = processor_dino(images=input_image_open, return_tensors="pt").to(devices)
                outputs1 = model_dino(**inputs1)
                image_features1 = outputs1.last_hidden_state
                image_features1 = image_features1.mean(dim=1)

                inputs2 = processor_dino(images=output_image_open, return_tensors="pt").to(devices)
                outputs2 = model_dino(**inputs2)
                image_features2 = outputs2.last_hidden_state
                image_features2 = image_features2.mean(dim=1)

                sim = cos(image_features1[0], image_features2[0]).item()
                sim = (sim + 1) / 2 

                clip_inputs1 = clip_processor(images=input_image_open, return_tensors="pt").to(devices)
                clip_image_features1 = clip_model.get_image_features(**clip_inputs1)

                clip_inputs2 = clip_processor(images=output_image_open, return_tensors="pt").to(devices)
                clip_image_features2 = clip_model.get_image_features(**clip_inputs2)
                clip_sim_i = cos(clip_image_features1[0],clip_image_features2[0]).item()

                inputs_prompt = tokenizer_clip(text_name_en, padding=True, return_tensors="pt").to(devices)
                text_features = clip_model.get_text_features(**inputs_prompt)
                clip_sim_t = cos(text_features[0],clip_image_features2[0]).item()

            if sim>0.95 and clip_sim_i>0.95:continue
            if sim<0.8 or clip_sim_i<0.65 or clip_sim_t<0.23:continue
            if sim>0.99 or clip_sim_i>0.99:continue

            xkey = "%09d" % index
            jsons["instruction"] = text_name_en
            jsons["instruction_zh"] = qwen_ins
            jsons["task"] = "subject"
            jsons["label"] = "kontext"
            jsons["dino"] = sim
            jsons["clipI"] = clip_sim_i
            jsons["clipT"] = clip_sim_t
            jsons["liqe_score"] = q_score_1
            jsons["liqe_score_edit"] = q_score_2
            jsons["liqe_score_clip"] = q_score_clip_1
            jsons["liqe_score_clip_edit"] = q_score_clip_2
            jsons["aesthetic_score_v2_5"] = aesthetic_lama_output_1
            jsons["aesthetic_score_v2_5_edit"] = aesthetic_lama_output_2
            jsons["shuttle_pro"] = shuttle_pro
            tmp = str([sim,clip_sim_i,clip_sim_t])
            sample = {
                "__key__": xkey, 
                "1.0.jpg": input_image_open,
                "2.jpg": output_image_open,
                "txt": qwen_ins + tmp,
                "json": jsons,
            }
            sink.write(sample)
            index += 1
        except Exception as e:
            print(str(e))
        
    sink.close()
        


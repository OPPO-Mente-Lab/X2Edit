# import paddle
# paddle.utils.run_check()
# from paddleocr import PaddleOCR

import sys, glob, os

import torch
from pathlib import Path
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
import cv2
import string

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

from run_inference import read_words_from_text, load_flux_pipeline, render_glyph_multi, run_inference_with_pipe

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



def qwenvl(images,model,processor,prompts,devices,images_final=None):
    if images:
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
    else:
        messages = [
            {
                "role": "user",
                "content": [
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

def resize(image_ori,size = 512):
    size_wh = image_ori.size
    if int(size_wh[0]*size/size_wh[1]) >= size:
        image_ori = transforms.Resize((size, int(size_wh[0]*size/size_wh[1])), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    else:
        image_ori = transforms.Resize((int(size_wh[1]*size/size_wh[0]),size), interpolation=transforms.InterpolationMode.BILINEAR)(image_ori)
    width, height = image_ori.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    image_ori = image_ori.resize((new_width, new_height))
    return image_ori

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
    if key.endswith(".text"):
        return key, value.read().decode("utf-8")

def filter_resolution(example):
    jpgs = example[".jpg"]
    if jpgs is None:
        return False
    if ".json" in example:
        jsons = example[".json"]
        # aesthetic_score = jsons["aesthetic_score"] #aesthetic_score
        # if aesthetic_score<5:
        #     return False
    return True

def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    txt = [example[".txt"] for example in examples]
    text = [example[".text"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json, "txt": txt, "text": text}


def is_tar_openable(filepath):
    try:
        with tarfile.open(filepath) as tar:
            return True
    except:
        print(f"Tar文件损坏或格式错误")
        return False



base_prompts = f'''做英文的改写：待修改的文字:"<text>"。 修改了之后的文字，内容和逻辑要合理，修改前后的字母长度接近。
        同时也要有变化，修改前后单词不能相同，含义可以相似也可以很大区别。例如"将'Exploration'改为'Multimodal'"。
        只需要生成一条编辑指令。
        '''

ocr_prompt = "直接输出图片的ocr内容，不要输出中间过程"

def remove_special_characters_and_lower(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    lower_case_text = cleaned_text.lower().replace(" ","")
    return lower_case_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--num_processors', default=8, type=int)
    parser.add_argument('--process_id', default=0, type=int)
    # parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/data_font_en/laion2b/*/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar
    # parser.add_argument('--file_root2', default="/mnt/data/group/text2img_data/data_font_en/laion400/*/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar
    # parser.add_argument('--file_root3', default="/mnt/data/group/text2img_data/data_font_en/coyo/*/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar
    # parser.add_argument('--file_root4', default="/mnt/data/group/text2img_data/data_font_en/BLIP_tar_512/*/*", type=str) # /mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar

    parser.add_argument('--file_root', default="/mnt/data/group/text2img_data/data_font_en/ae/*/*", type=str)

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors
    
    dtype = torch.bfloat16
    devices="cuda"

    pipe = load_flux_pipeline()

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
    # for tar in tqdm(glob.glob(args.file_root2)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    # for tar in tqdm(glob.glob(args.file_root3)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    # for tar in tqdm(glob.glob(args.file_root4)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    random.seed(1)
    random.shuffle(tar_list)

    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]
    print(len(tar_list))

    output_dir = f"/mnt/data/group/**/textflux-main/textflux-qwen-generate-en/{process_id+70}" 

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
                input_image_open = obj["jpg"][0]
                texts = obj["text"][0]
                try:
                    q_score_1 = iqa_metric(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0].tolist()
                    q_score_clip_1 = iqa_metric_clip(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0][0].tolist()
                    lama_output_tensor = preprocessor(images=input_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
                    aesthetic_lama_output_1 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()

                    if type(q_score_1)==list:
                        q_score_1=q_score_1[0]
                    if q_score_1<3 or aesthetic_lama_output_1<4 or q_score_clip_1<0.5:continue

                    width, height = input_image_open.size
                    direction = 'horizontal' if height > width else 'vertical'

                    chosen_mask = np.zeros((height, width), dtype=np.uint8)
                    chosen_text = ''
                    area = 0
                    for i, text in zip(range(len(jsons["font"])), texts.strip().replace('\n', '').split('&&')):
                        text = text.strip()
                        points_str = jsons["font"][i][1]
                        points = [tuple(map(int, point.split(','))) for point in points_str.replace("(", "").replace(")", "").split()]
                        
                        mask = np.zeros((height, width), dtype=np.uint8)
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask, [points], color=255)
                        if len(mask[mask > 0]) > area:
                            area = len(mask[mask > 0])
                            chosen_mask = mask
                            chosen_text = text
                    mask = chosen_mask
                    text = chosen_text
                    if len(mask[mask > 0]) / (width * height) < 0.01: continue

                    prompts = base_prompts.replace('<text>', text)
                    qwen_ins = qwenvl(None,model_7b,processor_7b,prompts,devices)[0]
                    
                    changed_text = re.findall(r'[\“"].*?[\”"]', qwen_ins)[-1][1:-1]
                    origin_text = re.findall(r'[\“"].*?[\”"]', qwen_ins)[-2][1:-1]
                    new_text = text.replace(origin_text, changed_text)

                    if remove_special_characters_and_lower(text)==remove_special_characters_and_lower(new_text):
                        continue
                    print(f"{text}->{new_text}")

                    mask_image = Image.fromarray(mask).convert("RGB")

                    input_image_open = resize(input_image_open)
                    mask_image = resize(mask_image)

                    rendered_text = render_glyph_multi(input_image_open, mask_image, [new_text])

                    if direction == 'horizontal':
                        # Horizontal concatenation [Rendered Image | Original Image]
                        combined_image = Image.fromarray(np.hstack((np.array(rendered_text), np.array(input_image_open))))
                        combined_mask = Image.fromarray(np.hstack((np.array(Image.new("RGB", input_image_open.size, (0, 0, 0))), np.array(mask_image))))
                    else:
                        # Vertical concatenation [Rendered Image / Original Image]
                        combined_image = Image.fromarray(np.vstack((np.array(rendered_text), np.array(input_image_open))))
                        combined_mask = Image.fromarray(np.vstack((np.array(Image.new("RGB", input_image_open.size, (0, 0, 0))), np.array(mask_image))))

                    t1 = time.time()
                    result, _ = run_inference_with_pipe(pipe, combined_image, combined_mask, new_text)
                    print("## textflux Time ##:",time.time()-t1)

                    width, height = result.size
                    if direction == 'horizontal':
                        output_image_open = result.crop((width // 2, 0, width, height))
                    else:
                        output_image_open = result.crop((0, height // 2, width, height))

                    q_score_2 = iqa_metric(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0].tolist()
                    q_score_clip_2 = iqa_metric_clip(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0][0].tolist()
                    lama_output_tensor = preprocessor(images=output_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
                    aesthetic_lama_output_2 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()
                    if type(q_score_2)==list:
                        q_score_2=q_score_2[0]
                    if q_score_2<3.5 or aesthetic_lama_output_2<4.5 or q_score_clip_2<0.6:continue

                    # output_image_np = np.array(output_image_open)
                    # _, rec_res, _ = ocr(output_image_np, cls=False)
                    # detectd_text = [text for text, _ in rec_res]

                    detectd_text = qwenvl(output_image_open,model_7b,processor_7b,ocr_prompt,devices)[0]

                    print(f"detectd_text:{detectd_text}")
                    if new_text not in detectd_text: 
                        continue
                        
                    _SC_prompt = SC_prompts.replace("<edit_prompt>", qwen_ins)
                    qwen_ins_result = qwenvl_score(input_image_open,output_image_open,model,processor,_SC_prompt,devices)[0]
                    scores = re.findall(r'Score \d: (\d+)', qwen_ins_result)
                    scores = [int(score) for score in scores]
                    
                    xkey = "%09d" % index
                    jsons["instruction"] = qwen_ins
                    jsons["label"] = "textflux"
                    jsons["task"] = "text change"
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
                        ".mask.jpg": mask_image,
                        "txt": str(qwen_ins) + "\n" + str(qwen_ins_result) + tmp,
                        "json": jsons,
                    }
                    sink.write(sample)
                    index += 1
                except Exception as e:
                    print(str(e))
                    continue
                
    sink.close()
        


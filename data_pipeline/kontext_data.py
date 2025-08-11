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
import io
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import requests
import re
from collections import deque, defaultdict
import random
from diffusers import FluxKontextPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
import gc
import pyiqa
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip


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



def find_closest_aspect_bucket(image_size):
    image_height, image_width = image_size
    image_aspect = image_width / image_height
    aspect_diffs = [abs((bucket[0] / bucket[1]) - image_aspect) for bucket in BUCKETS]
    closest_bucket_index = np.argmin(aspect_diffs)
    return BUCKETS[closest_bucket_index]


def resize_to_bucket(image, output_path=None):

    target_h, target_w = find_closest_aspect_bucket(image.size)
    
    transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.CenterCrop((target_h, target_w))
    ])
    
    processed = transform(image)
    
    if output_path:
        torchvision.io.write_image(processed, output_path)
    
    return processed

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(text)
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
    # if jpg[0]/jpg[1] > 2.1 or jpg[0]/jpg[1]<0.48:
    #     return False
    if jpg[0]*jpg[1]<900*900:
        return False
    if ".json" in example:
        jsons = example[".json"]
        aesthetic_score = jsons["aesthetic_score"] #aesthetic_score
        if aesthetic_score<5.5:
            return False
    return True


def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[".jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    return {"jpg": jpg, "key": key,"json": json}

def get_random_items(d, n):
    return dict(random.sample(list(d.items()), min(n, len(d))))

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



rules_ori = '''
{
  "相机移动": "...",
  "材料更改": "...",
  "添加主体": "...",
  "删除主体": "...",
  "替换主体": "...",
  "色调变换": "...",
  "风格更改": "...",
  "复杂推理": "...",
  "主题生成": "...",
  "证件照生成": "..."
}
'''

prompts = f'''参考图片的内容，给出合理的图片编辑指令的prompts是什么？根据图片看适合做什么指令, 
        每个类型对应1条具体指令，输出内容的多样性更多一些。针对复杂推理任务，指令的意图必须是基于参考图片的内容事实，以编辑图片作画为目的,一般是一个问句的形式。比如
        你能提供这辆车的内饰图片吗？ ，有没有与这些衬衫的设计相匹配的手机壳？你能展示一些吗？；针对主题生成任务，一般就是参考图片的风格或者主体内容，再另一个场景生成另一幅画。
        针对相机移动，以展现场景的新视角。根据场景提供差异显著的相机运动类型（例如：相机现在俯拍房间；人物侧面肖像视角，是否聚焦图像主要主体，提供不同程度的变焦，拉进或者推远镜头等）；
        针对色调变换，为图像建议新的灯光设置，提出各种灯光舞台和设置、不同时间段的光线、移除或添加新的自然光等，如果图片不清晰，可以使其变清晰，修复旧图片，上色，去模糊等等。
        编辑指令的类型和输出格式请参考：{rules_ori}。只输出结果即可。
        如果图片不适合做相应的指令类型，值输出空即可,不需要给理由'''




if __name__ == "__main__":
    devices = "cuda"
    dtype = torch.bfloat16

    path = "/mnt/workspace/group/models/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2").to(device=devices)
    processor = AutoProcessor.from_pretrained(path)

    pipe = FluxKontextPipeline.from_pretrained("/mnt/workspace/group/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to(devices)

    tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/group/models/opus-mt-zh-en")
    model_trans = AutoModelForSeq2SeqLM.from_pretrained("/mnt/workspace/group/models/opus-mt-zh-en").to(devices)

    ## 
    path = "/mnt/workspace/group/text2img_data/X2I/ImgEdit/ImgEdit_Judge"
    model_score = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        local_files_only=True,
    ).to(devices)
    min_pixels = 1016064  # we train our model with this settings
    max_pixels = 1354752  # we train our model with this settings
    processor_score = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)  

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
    parser.add_argument('--num_processors', default=32, type=int)
    parser.add_argument('--process_id', default=1, type=int)
    parser.add_argument('--file_root', default="/mnt/workspace/group/text2img_data/suoping_en_tar/*/*/*tar", type=str) # /mnt/workspace/group/text2img_data/data_process/coyo1/{00000..51975}.tar
    parser.add_argument('--file_root0', default="/mnt/workspace/group/text2img_data/data_process/data_scraping_2023/*tar", type=str) 
    parser.add_argument('--file_root1', default="/mnt/workspace/group/text2img_data/data_process/coyo1/*", type=str) 
    parser.add_argument('--file_root2', default="/mnt/workspace/group/text2img_data/suoping_zh_tar/*/*tar", type=str) 
    parser.add_argument('--file_root3', default="/mnt/workspace/group/text2img_data/wall_9W/*.tar", type=str) 

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors


    tar_list = []
    for tar in tqdm(glob.glob(args.file_root)):
        if tar.endswith(".tar"):
            tar_list.append(tar)
    # for tar in tqdm(glob.glob(args.file_root0)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    # for tar in tqdm(glob.glob(args.file_root1)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    for tar in tqdm(glob.glob(args.file_root2)):
        if tar.endswith(".tar"):
            tar_list.append(tar)
    # for tar in tqdm(glob.glob(args.file_root3)):
    #     if tar.endswith(".tar"):
    #         tar_list.append(tar)
    random.seed(1)
    random.shuffle(tar_list)
    # tar_list = tar_list[::-1]

    print(len(tar_list))

    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]


    output_dir = f"/mnt/workspace/group/text2img_data/X2I/tars/Kontext_aspect1/{process_id+32}" 


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
        
            input_image_open = resize_to_bucket(jpg)
            q_score_1 = iqa_metric(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0].tolist()
            if type(q_score_1)==list:
                q_score_1=q_score_1[0]
            q_score_clip_1 = iqa_metric_clip(transforms.ToTensor()(input_image_open).unsqueeze(0).to(devices))[0][0].tolist()
            lama_output_tensor = preprocessor(images=input_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
            aesthetic_lama_output_1 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()
            # if q_score_1<3 or aesthetic_lama_output_1<4 or q_score_clip_1<0.55:continue

            try:
                # with torch.no_grad():
                qwen_ins_result = qwenvl(input_image_open,model,processor,prompts,devices)[0]
                if "```json" in qwen_ins_result:
                    matches = re.findall(r"```json(.*?)```", qwen_ins_result, re.DOTALL)
                    qwen_ins_result = matches[0].strip()

                qwen_ins_dict = json.loads(qwen_ins_result)
                if not qwen_ins_dict: continue
                
                qwen_ins_dict = {k: v for k, v in qwen_ins_dict.items() if v not in [None, '']}
                qwen_ins_dict = get_random_items(qwen_ins_dict,2)

                for qwen_ins_type, qwen_ins in tqdm(qwen_ins_dict.items()):
                    if not contains_chinese(qwen_ins):continue
                    text_name_en = translate(qwen_ins)[0]

                    output_image_open = pipe(
                        image=input_image_open,
                        prompt=text_name_en,
                        guidance_scale=2.5,
                        height=input_image_open.size[1],
                        width=input_image_open.size[0],
                    ).images[0]
                    

                    q_score_2 = iqa_metric(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0].tolist()
                    if type(q_score_2)==list:
                        q_score_2=q_score_2[0]
                    q_score_clip_2 = iqa_metric_clip(transforms.ToTensor()(output_image_open).unsqueeze(0).to(devices))[0][0].tolist()
                    lama_output_tensor = preprocessor(images=output_image_open, return_tensors="pt").pixel_values.to(torch.bfloat16).to(devices)
                    aesthetic_lama_output_2 = model_ae(lama_output_tensor).logits.squeeze().float().tolist()

                    # if q_score_2<3.5 or aesthetic_lama_output_2<4.5 or q_score_clip_2<0.6:continue    
                    if abs(q_score_2-q_score_1)<0.02 or abs(q_score_clip_1-q_score_clip_2)<0.008 or abs(aesthetic_lama_output_1-aesthetic_lama_output_2)<0.02:continue    

                    # gc.collect()
                    # torch.cuda.empty_cache()
                    _SC_prompt = SC_prompts.replace("<edit_prompt>", qwen_ins)
                    qwen_ins_result = qwenvl_score(input_image_open,output_image_open,model_score,processor_score,_SC_prompt,devices)[0]
                    scores = re.findall(r'Score \d: (\d+)', qwen_ins_result)
                    scores = [int(score) for score in scores]
                    if sum(scores)<4:continue

                    xkey = "%09d" % index
                    jsons["instruction"] = text_name_en
                    jsons["instruction_zh"] = qwen_ins
                    jsons["task"] = qwen_ins_type
                    jsons["label"] = "Kontext"
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
        


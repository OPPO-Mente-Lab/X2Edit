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
from torchvision import transforms
from torchdata.datapipes.iter import FileOpener
import requests
import random
import tarfile

import re
from collections import deque, defaultdict
import random

def extract_numbers(text):
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if match:
        num1, num2 = int(match.group(1)), int(match.group(2))
        return [num1, num2]
    else:
        return None

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

def qwenvl(img_url0,img_url1,model,processor,prompts,devices):
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

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
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

def collate_fn(examples):
    key = [example["__key__"].split("/")[-1] for example in examples]
    jpg1 = [example[".1.jpg"] for example in examples]
    jpg2 = [example[".2.jpg"] for example in examples]
    jpg3 = [example[".3.jpg"] for example in examples]
    json = [example[".json"] for example in examples]
    txt = [example[".txt"] for example in examples]
    return {"jpg1": jpg1,"jpg2": jpg2,"jpg3": jpg3,"txt": txt, "key": key, "json": json}

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

    path = "/mnt/data/group/text2img_data/X2I/ImgEdit/ImgEdit_Judge"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        local_files_only=True,
    ).to(devices)
    processor = AutoProcessor.from_pretrained(path)

    parser = argparse.ArgumentParser("Data Processing Pipeline", add_help=True)
    parser.add_argument('--file_root', default="/mnt/data/group/**/gendata/outputs_flux/*/*", type=str)
    parser.add_argument('--process_id', default=1, type=int)
    parser.add_argument('--num_processors', default=24, type=int)

    args = parser.parse_args()
    process_id = args.process_id
    num_processors = args.num_processors

    tar_list = []
    for tar in tqdm(glob.glob(args.file_root)):
        if tar.endswith(".tar"):
            tar_list.append(tar)


    curr_tar_lists = np.array_split(tar_list, num_processors)[process_id]
    print(len(tar_list))

    output_dir = f"/mnt/data/group/text2img_data/X2I/tars/delete/{process_id}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index = 0
    num_per_dir = 5000
    pattern = os.path.join(output_dir, f"%05d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(6e9), maxcount=int(num_per_dir))
    
    rs = MultiProcessingReadingService(num_workers=1)
    for tar_name in curr_tar_lists:
        if is_tar_openable(tar_name):
            dataset = FileOpener([tar_name], mode="b").load_from_tar().map(decode).webdataset(). \
                    batch(1).collate(collate_fn=collate_fn)
            dl = DataLoader2(dataset, reading_service=rs)
            try:   
                for obj in tqdm(dl):
                    for i in range(len(obj["json"])):
                        try:
                            _SC_prompt = SC_prompts.replace("<edit_prompt>", obj["json"][i]["instruction_zh"])
                            qwen_ins_result = qwenvl(obj["jpg1"][i],obj["jpg2"][i],model,processor,_SC_prompt,devices)[0]
                            scores = re.findall(r'Score \d: (\d+)', qwen_ins_result)
                            scores = [int(score) for score in scores]
                            new_json = obj["json"][i]
                            new_json["score"] = str(scores)
                            sample = {
                                "__key__": obj["key"][i], 
                                ".1.0.jpg": obj["jpg1"][i],
                                ".2.jpg": obj["jpg2"][i],
                                "txt": obj["json"][i]["instruction_zh"] + '\n' + qwen_ins_result,
                                "json": new_json,
                            }
                            sink.write(sample)
                            index += 1
                        except Exception as e:
                            print(str(e))
            except Exception as e:
                print(str(e))
    
    sink.close()
        


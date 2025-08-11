import json
import requests
import hmac
import hashlib
from collections import OrderedDict
import time
import base64
from PIL import Image
from io import BytesIO
import os 
import torch
from tqdm import tqdm
import argparse
from viescore import VIEScore
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import ast
from functools import wraps
import signal

prompt = """
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
"""

import signal

# 1. 定义一个超时处理函数
def handler(signum, frame):
    raise TimeoutError("执行超过30秒!")

# 2. 将SIGALRM信号与处理函数关联
signal.signal(signal.SIGALRM, handler)

class score_pipeline:
    def __init__(self, backbone):
        self.backbone = backbone
        assert backbone in ["gpt4o", "ImgEditJudge", "qwen25vl"]

        if backbone == "ImgEditJudge":
            # Load the model with recommended configurations
            self.prompt = prompt
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                # "/mnt/workspace/*****/Qwen2.5-VL-7B-Instruct",
                "/mnt/data/group/text2img_data/X2I/ImgEdit/ImgEdit_Judge",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cuda",
                local_files_only=True,
            )
            min_pixels = 1016064  # we train our model with this settings
            max_pixels = 1354752  # we train our model with this settings
            self.processor = AutoProcessor.from_pretrained("/mnt/data/group/text2img_data/X2I/ImgEdit/ImgEdit_Judge", min_pixels=min_pixels, max_pixels=max_pixels) 
        else:
            self.model = VIEScore(backbone=backbone, task="tie", key_path='secret_t2.env',device="cuda")
    
    def score(self, img_url0, img_url1, instruction, output_txt):
        if self.backbone == "ImgEditJudge":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt.replace("<edit_prompt>", instruction)},
                        {"type": "image", "image": img_url0},
                        {"type": "image", "image": img_url1},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            score_list = []
            for line_text in output_text.split('\n'):
                if line_text.strip() == '': continue
                key, value = line_text.split(': ')
                value = value.split(' (')[0]
                if 'Score' in key or 'score' in key:
                    score_list.append(int(value))
            print(instruction, output_text)

            with open(output_txt, "a", encoding="utf-8") as fw:
                fw.write(instruction + "\t" + str(score_list)+"\n")
            
            return score_list
        elif self.backbone == "gpt4o":
            try_times = 0
            while try_times < 3:
                try:
                    signal.alarm(30)

                    score_list = self.model.evaluate([img_url0, img_url1], instruction)
                    print(instruction, score_list)
                    with open(output_txt, "a", encoding="utf-8") as fw:
                        fw.write(instruction + "\t" + str(score_list)+"\n")
                    
                    return score_list
                except TimeoutError as e:
                    print(f"发生错误：{e}，\n img_url0:{img_url0}\n img_url1:{img_url1}\n instruction:{instruction}\n 正在重新尝试...")
                    time.sleep(1)
                    try_times += 1
                except Exception as e:
                    print(f"发生错误：{e}，\n img_url0:{img_url0}\n img_url1:{img_url1}\n instruction:{instruction}\n 正在重新尝试...")
                    time.sleep(1)
                    try_times += 1
                finally:
                    signal.alarm(0)  # 重置定时器
        else:
            score_list = self.model.evaluate([img_url0, img_url1], instruction)
            print(instruction, score_list)
            with open(output_txt, "a", encoding="utf-8") as fw:
                fw.write(instruction + "\t" + str(score_list)+"\n")
            
            return score_list
        
def has_subfolders(folder_path):
    return any(os.path.isdir(os.path.join(folder_path, item)) for item in os.listdir(folder_path))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backnone', type=str, default="gpt4o")
    parser.add_argument('--ref_dir', type=str, default="/mnt/data/group/****/AnySD/outputs_reasoning_bench_AnySD")
    parser.add_argument('--in_suffix', type=str, default="_in.jpg")
    parser.add_argument('--out_suffix', type=str, default="_out.jpg")
    parser.add_argument('--output_txt_dir', type=str, default="/mnt/data/group/****/gpt4o-test")
    args = parser.parse_args()

    ref_dir = args.ref_dir
    output_txt_dir = args.output_txt_dir
    backbone = args.backnone
    out_suffix = args.out_suffix
    in_suffix = args.in_suffix

    os.makedirs(output_txt_dir, exist_ok=True)
    txt_name = ref_dir.split('/')[-2] + ':' + ref_dir.split('/')[-1] + out_suffix.split('.jpg')[0]
    output_txt = os.path.join(output_txt_dir, txt_name + '.txt')

    pipeline = score_pipeline(backbone)

    scores=[]
    instructions_score = {}
    if os.path.exists(output_txt):
        with open(output_txt, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                if 'final score' in line or '\t' not in line:continue
                text_part, list_part = line.strip().split('\t')
                extracted_list = ast.literal_eval(list_part)
                score_list = [float(x) for x in extracted_list]
                instructions_score[text_part] = score_list
                scores.append(score_list)

    score_tasks = {}
    if has_subfolders(ref_dir):
        for path_1 in os.listdir(ref_dir):
            score_task_list = []
            output_path0 = os.path.join(ref_dir,path_1)
            if not os.path.isdir(output_path0): continue
            for filename in tqdm(os.listdir(output_path0)):
                if out_suffix not in filename:continue
                img_url1 = os.path.join(output_path0, filename)
                img_url0 = img_url1.replace(out_suffix,in_suffix)
                instruction = filename.replace(out_suffix,"")
                if instruction in instructions_score.keys():
                    print(instruction)
                    score_task_list.append(instructions_score[instruction])
                    continue
                try:
                    score_list = pipeline.score(img_url0, img_url1, instruction, output_txt)
                    # scores.append(sum(score_list)/len(score_list))
                    if score_list is not None:
                        scores.append(score_list)
                    score_task_list.append(score_list)
                except Exception as e:
                    print(instruction)
                    print(str(e))
                
            try:
                score_tasks[path_1] = [sum(column) / len(column) for column in zip(*score_task_list)]
                if len(score_tasks[path_1]) == 2:
                    score_tasks[path_1].append(round(score_tasks[path_1][0]+score_tasks[path_1][1],3))
                print(f'{path_1}:{score_tasks[path_1]}')
            except Exception as e:
                print(str(e))
    else:
        output_path0 = ref_dir
        for filename in os.listdir(output_path0):
            if out_suffix not in filename:continue
            img_url1 = os.path.join(output_path0, filename)
            img_url0 = img_url1.replace(out_suffix,in_suffix)
            instruction = filename.replace(out_suffix,"")
            if instruction in instructions_score.keys():
                print(instruction)
                continue

            try:
                score_list = pipeline.score(img_url0, img_url1, instruction, output_txt)
                # scores.append(sum(score_list)/len(score_list))
                if score_list is not None:
                    scores.append(score_list)
            except Exception as e:
                print(instruction)
                print(str(e))

    averages = [sum(column) / len(column) for column in zip(*scores)]
    if len(averages) == 2:
        averages.append(round(averages[0]+averages[1],3))
    with open(output_txt, "a", encoding="utf-8") as fw:
        fw.write("final score: " + str(averages)+"\n")
    
    try:
        for k, v in score_tasks.items():
            with open(output_txt, "a", encoding="utf-8") as fw:
                fw.write(f"{k}: {v}\n")
    except Exception as e:
        print(str(e))
    
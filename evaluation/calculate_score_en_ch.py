import ast
import glob
import re
import string
from tqdm import tqdm

def contains_chinese(string):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(string))

if __name__ == "__main__":

    en_score_list = []
    ch_score_list = []

    # 打开文件
    txt_paths = [
        '/mnt/data/group/**/step1x-test-score/gedit-txt/gpt4o/outputs_gedit:all_moe_task_dispersive_1024_0721_out5100.txt',
        '/mnt/data/group/**/step1x-test-score/reasoning-bench-txt/gpt4o/outputs_gedit:reason_out13500.txt'
    ]

    for txt_path in txt_paths:
        text_list = []
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 逐行读取
            for line in tqdm(file):
                if 'final score' in line or '\t' not in line or len(line.strip().split('\t')) != 2:continue
                text_part, list_part = line.strip().split('\t')
                if text_part in text_list:continue
                text_list.append(text_part)
                # 提取列表字符串并转换为实际列表
                extracted_list = ast.literal_eval(list_part)

                # 将列表元素转换为得分
                score_list = [float(x) for x in extracted_list]

                if ('A' <= text_part[0] <= 'Z' or 'a' <= text_part[0] <= 'z') or not contains_chinese(text_part):
                    en_score_list.append(score_list)
                else:
                    ch_score_list.append(score_list)

    if ch_score_list:
        ch_score_mean_score = [round(sum(column) / len(column),3) for column in zip(*ch_score_list)]
        if len(ch_score_mean_score) == 2:
            ch_score_mean_score.append(round(ch_score_mean_score[0]+ch_score_mean_score[1],3))
        print(f'ch_score_mean_score:{ch_score_mean_score}')
    
    if en_score_list:
        en_score_mean_score = [round(sum(column) / len(column),3) for column in zip(*en_score_list)]
        if len(en_score_mean_score) == 2:
            en_score_mean_score.append(round(en_score_mean_score[0]+en_score_mean_score[1],3))
        print(f'en_score_mean_score:{en_score_mean_score}')

    total_mean_score = []
    total_score_list = en_score_list + ch_score_list
    total_mean_score = [round(sum(column) / len(column),3) for column in zip(*total_score_list)]
    if len(total_mean_score) == 2:
        total_mean_score.append(round(total_mean_score[0]+total_mean_score[1],3))
    print(f'total_mean_score:{total_mean_score}')

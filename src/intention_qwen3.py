import time
import torch 
import os

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

def qwen3(prompt,model,tokenizer):
    #        其中左右移动物体，放大或者缩小物体是属于相机移动的编辑类型；
    ins = f"""
        我现在输入一个图片编辑指令:{prompt}，总共15个类别:{list(all_dict.values())}，判断我的指令属于哪个类别。直接输出类别，不需要中间结果; 
        其中implicit_change表示推理的问题，比如**情况，发生什么？或者改变物体之间关系; low_leval表示图片转换，比如Canny/Depth/Linear image generates images； 
        subject表示没有明确的编辑意图，而是对图片的主体做了其他描述的个性化生成,比如A sticker of this object。
    """

    messages = [
        {"role": "user", "content":ins}
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

if __name__ == "__main__":
    # prepare the model input
    # prompt = '将她的卷发变成直发'

    while True:
        prompt = input("\nPlease Input Query (stop to exit) >>> ")
        if not prompt:
            print('Query should not be empty!')
            continue
        if prompt == "stop":
            break

        t1 = time.time()
        content = qwen3(prompt)
        print("time:", time.time()-t1)
        print("content:", content)

    # a=b=0
    # for paths in os.listdir("/mnt/workspace/group/******/AndesDiT/EasyControl/outputs_gpt4o_high/ps_human"):
    #     if "_in.jpg" not in paths:continue
    #     prompt = paths.replace("_in.jpg","")
    #     t1 = time.time()
    #     content = qwen3(prompt)
    #     a+=1
    #     if content=="人物修图":
    #         b+=1
    #     print(f"time:{time.time()-t1}, prompt:{prompt},result:{content}")
    # print(b/a)
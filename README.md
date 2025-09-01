<div align="center">
  <h1>X2Edit</h1>
<a href='https://arxiv.org/abs/2508.07607'><img src='https://img.shields.io/badge/arXiv-2508.07607-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/datasets/OPPOer/X2Edit-Dataset'><img src='https://img.shields.io/badge/🤗%20HuggingFace-X2Edit Dataset-ffd21f.svg'></a>
<a href='https://huggingface.co/OPPOer/X2Edit'><img src='https://img.shields.io/badge/🤗%20HuggingFace-X2Edit-ffd21f.svg'></a>
<a href='https://www.modelscope.cn/datasets/AIGCer-OPPO/X2Edit-Dataset'><img src='https://img.shields.io/badge/🤖%20ModelScope-X2Edit Dataset-purple.svg'></a>
</div>

> **X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning**
> <br>
[Jian Ma](https://scholar.google.com/citations?hl=zh-CN&user=XtzIT8UAAAAJ)<sup>1</sup>*, 
[Xujie Zhu](https://github.com/CVPIE)<sup>2</sup>*,
[Zihao Pan](https://scholar.google.com.hk/citations?user=tXlKGqQAAAAJ&hl=zh-CN)<sup>2</sup>*,
[Qirong Peng](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=gUPpazEAAAAJ)<sup>1</sup>*, 
[Xu Guo](https://github.com/Guoxu1233)<sup>3</sup>, 
[Chen Chen](https://scholar.google.com/citations?user=CANDhfAAAAAJ&hl=zh-CN)<sup>1</sup>,
[Haonan Lu](https://scholar.google.com/citations?user=EPBgKu0AAAAJ&hl=en)<sup>1</sup>
<br>
<sup>1</sup>OPPO AI Center, <sup>2</sup>Sun Yat-sen University, <sup>3</sup>Tsinghua University
<br>


## X2Edit image generation results
<div align="center">
  <img src="assets/X2Edit images.jpg">
</div>


## News
- 2025/08/25 Support **[Qwen-Image](https://github.com/QwenLM/Qwen-Imag)** for training and inference. **[Checkpoint](https://huggingface.co/OPPOer/X2Edit/tree/main/model_qwen_image)**

### X2Edit image generation results with Qwen-Image
<div align="center">
  <img src="assets/qwen-image1.png">
</div>
<div align="center">
  <img src="assets/qwen-image0.png">
</div>


## Environment

Prepare the environment, install the required libraries:

```shell
$ cd X2Edit
$ conda create --name X2Edit python==3.11
$ conda activate X2Edit
$ pip install -r requirements.txt
```

Clone **[LaMa](https://github.com/advimman/lama.git)** to [`data_pipeline`](./data_pipeline) and rename it to [`lama`](./data_pipeline/lama). Clone **[SAM](https://github.com/facebookresearch/segment-anything.git)** and **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO.git)** to [`SAM`](./data_pipeline/SAM), and then rename them to [`segment_anything`](./data_pipeline/SAM/segment_anything) and [`GroundingDINO`](./data_pipeline/SAM/GroundingDINO)


## Data Construction
(./assets/dataset_detail.jpg)

X2Edit provides executable scripts for each data construction workflow shown in the figure. We organize the dataset using the **[WebDataset](https://github.com/webdataset/webdataset)** format. Please replace the dataset in the scripts. The following Qwen model can be selected from **[Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)**, **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**, and **[Qwen2.5-VL-7B](https://huggingface.co/datasets/sysuyy/ImgEdit/tree/main/ImgEdit_Judge)**. In addition, we also use aesthetic scoring models for screening, please donwload **[SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)** and **[aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth)**, and then change the path in [`siglip_v2_5.py`](./data_pipeline/aesthetic_predictor_v2_5/siglip_v2_5.py).

- **Subject Addition & Deletion** → use [`expert_subject_deletion.py`](./data_pipeline/expert_subject_deletion.py) and [`expert_subject_deletion_filter.py`](./data_pipeline/expert_subject_deletion_filter.py): The former script is used to construct deletion-type data, while the latter uses the fine-tuned **Qwen2.5-VL-7B** to further screen the constructed deletion-type data. Before executing, download **[RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)**, **[GroundingDINO](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth)**, **[SAM](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth)**, **[Randeng-Deltalm](https://huggingface.co/IDEA-CCNL/Randeng-Deltalm-362M-Zh-En)**, **[InfoXLM](https://huggingface.co/microsoft/infoxlm-base)**, **[RMBG](https://huggingface.co/briaai/RMBG-2.0)** and **[LaMa](https://huggingface.co/smartywu/big-lama)**.
- **Normal Editing Tasks** → use [`step1x_data.py`](./data_pipeline/step1x_data.py): Please download the checkpoint **[Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)**. The language model we use is **Qwen2.5-VL-72B**.
- **Subject-Driven Generation** → use [`kontext_subject_data.py`](./data_pipeline/kontext_subject_data.py): Please download the checkpoints **[FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)**, **[DINOv2](https://huggingface.co/facebook/dinov2-giant)**, **[CLIP](https://huggingface.co/openai/clip-vit-large-patch14)**, **[OPUS-MT-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)**, **[shuttle-3-diffusion](https://huggingface.co/shuttleai/shuttle-3-diffusion)**. The language model we use is **Qwen3-8B**.
- **Style Transfer** → use [`kontext_style_transfer.py`](./data_pipeline/kontext_style_transfer.py): Please download the checkpoints **[FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)**, **[DINOv2](https://huggingface.co/facebook/dinov2-giant)**, **[CLIP](https://huggingface.co/openai/clip-vit-large-patch14)**, **[OPUS-MT-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)**, **[shuttle-3-diffusion](https://huggingface.co/shuttleai/shuttle-3-diffusion)**. The language model we use is **Qwen3-8B**.
- **Style Change** → use [`expert_style_change.py`](./data_pipeline/expert_style_change.py): Please download the checkpoints **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**, **[OmniConsistency](https://huggingface.co/showlab/OmniConsistency)**. We use **Qwen2.5-VL-7B** to score.
- **Text Change** → use [`expert_text_change_ch.py`](./data_pipeline/expert_text_change_ch.py) for **Chinese** and use [`expert_text_change_en.py`](./data_pipeline/expert_text_change_en.py) for **English**: Please download the checkpoint **[textflux](https://huggingface.co/yyyyyxie/textflux)**. We use **Qwen2.5-VL-7B** to score.
- **Complex Editing Tasks** → use [`bagel_data.py`](./data_pipeline/bagel_data.py): Please download the checkpoint **[Bagel](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)**. We use **Qwen2.5-VL-7B** to score.
- **High Fidelity Editing Tasks** → use [`gpt4o_data.py`](./data_pipeline/gpt4o_data.py): Please download the checkpoint **[OPUS-MT-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)** and use your own GPT-4o API. We use **Qwen2.5-VL-7B** to score.
- **High Resoluton Data Construction** → use [`kontext_data.py`](./data_pipeline/kontext_data.py): Please download the checkpoint **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)** and **[OPUS-MT-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)**. We use **Qwen2.5-VL-7B** to score.

## Inference
We provides inference scripts for editing images with resolutions of **1024** and **512**. In addition, we can choose the base model of X2Edit, including **[FLUX.1-Krea](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)**, **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**, **[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)**, **[PixelWave](https://huggingface.co/mikeyandfriends/PixelWave_FLUX.1-dev_03)**, **[shuttle-3-diffusion](https://huggingface.co/shuttleai/shuttle-3-diffusion)**, and choose the LoRA for integration with MoE-LoRA including **[Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)**, **[AntiBlur](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur)**, **[Midjourney-Mix2](https://huggingface.co/strangerzonehf/Flux-Midjourney-Mix2-LoRA)**, **[Super-Realism](https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA)**, **[Chatgpt-Ghibli](https://huggingface.co/openfree/flux-chatgpt-ghibli-lora)**. Choose the model you like and download it. For the MoE-LoRA, we will open source a unified checkpoint that can be used for both 512 and 1024 resolutions. 

Before executing the script, download **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** to select the task type for the input instruction, base model(**FLUX.1-Krea**, **FLUX.1-dev**, **FLUX.1-schnell**, **shuttle-3-diffusion**), **[MLLM](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** and **[Alignet](https://huggingface.co/OPPOer/X2I/blob/main/qwen2.5-vl-7b_proj.pt)**. All scripts follow analogous command patterns. Simply replace the script filename while maintaining consistent parameter configurations.

```shell
$ python infer.py --device cuda --pixel 1024 --num_experts 12 --base_path BASE_PATH --qwen_path QWEN_PATH --lora_path LORA_PATH --extra_lora_path EXTRA_LORA_PATH
$ python infer_qwen.py --device cuda --pixel 1024 --num_experts 12 --base_path BASE_PATH --qwen_path QWEN_PATH --lora_path LORA_PATH --extra_lora_path EXTRA_LORA_PATH  ## for Qwen-Image backbone

```

**device:** The device used for inference. default: `cuda`<br>
**pixel:** The resolution of the input image, , you can choose from **[512, 1024]**. default: `1024`<br>
**num_experts:** The number of expert in MoE. default: `12`<br>
**base_path:** The path of base model.<br>
**qwen_path:** The path of model used to select the task type for the input instruction. We use **Qwen3-8B** here.<br>
**lora_path:** The path of MoE-LoRA in X2Edit.<br>
**extra_lora_path:** The path of extra LoRA for plug-and-play. default: `None`.<br>

## Train
We organize the dataset using the **[WebDataset](https://github.com/webdataset/webdataset)** format. 
Please replace the dataset in the training script. Before executing, download **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)**, **[MLLM](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** and **[Alignet](https://huggingface.co/OPPOer/X2I/blob/main/qwen2.5-vl-7b_proj.pt)**, replace the paths  in [`train_1024.sh`](./train_1024.sh) and [`train_512.sh`](./train_512.sh).
Then you can run:

   - **For 1024 resolution**  
     ```shell
     bash training_script/train_1024.sh
     bash training_script/train_qwen_x2edit_1024.sh  ## for Qwen-Image backbone
     ```

   - **For 512 resolution**  
     ```shell
     bash training_script/train_512.sh
     bash training_script/train_qwen_x2edit_512.sh  ## for Qwen-Image backbone
     ```
**Important parameters in script**<br>
**rank:** The rank in LoRA. default: `64`<br>
**experts:** The number of expert in MoE. default: `12`<br>
**task:** The number of editing task. default: `16`<br>
**X2I_MLLM:** The checkpoint path of **MLLM**.<br>
**X2I_alignet:** The checkpoint path of **Alignet**.<br>

## Evaluation
We provides evaluation scripts for the calculation of VIEScore and ImgEdit-Judge-Score. Ensure that the edited images and the original images are placed in the same folder and named using the format "instruction + suffix". For example:
```shell
original image: Change the background to the sea._in.jpg
edited image: Change the background to the sea._out.jpg
```
in this case, instruction is "Change the background to the sea", suffix of original image is "_in.jpg", suffix of edited image is "_out.jpg".

- **VIEScore(for gpt4o or qwen25vl)**:  
```shell
$ cd evaluation
$ python score_i2i_new.py --backnone gpt4o --in_suffix _in.jpg --out_suffix _out.jpg --ref_dir YOUR_DATA_PATH --output_txt_dir ./score
```

- **ImgEdit-Judge-Score**:
```shell
$ cd evaluation
$ python score_i2i_new.py --backnone ImgEditJudge --in_suffix _in.jpg --out_suffix _out.jpg --ref_dir YOUR_DATA_PATH --output_txt_dir ./score
```

**backnone:** Model for evaluation, you can choose from **['gpt4o', 'qwen25vl', 'ImgEditJudge']**. For **qwen25vl**, download **[Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)** and change the path in [`__init__.py`](./evaluation/viescore/__init__.py). For **ImgEditJudge**, download the fine-tuned **[Qwen2.5-VL-7B](https://huggingface.co/datasets/sysuyy/ImgEdit/tree/main/ImgEdit_Judge)**.<br>
**in_suffix:** The suffix of the original image file name.<br>
**out_suffix:** The suffix of the edited image file name.<br>
**ref_dir:** The folder of edited images and original images.<br>
**output_txt_dir:** The folder for file used to record scores.<br>

- **Calculate Average Score**:
```shell
$ cd evaluation
$ python calculate_score_en_ch.py
```
## Acknowledgements 

This code is built on the code from the [diffusers](https://github.com/huggingface/diffusers) and [EasyControl](https://github.com/Xiaojiu-z/EasyControl).


## Citation

🌟 If you find our work helpful, please consider citing our paper and leaving valuable stars

```
@misc{ma2025x2editrevisitingarbitraryinstructionimage,
      title={X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning}, 
      author={Jian Ma and Xujie Zhu and Zihao Pan and Qirong Peng and Xu Guo and Chen Chen and Haonan Lu},
      year={2025},
      eprint={2508.07607},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.07607}, 
}
```





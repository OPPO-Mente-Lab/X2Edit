import argparse
import os,glob
os.system("pip install transformers==4.48")

import sys
import json
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

import SAM.GroundingDINO.groundingdino.datasets.transforms as T
from SAM.GroundingDINO.groundingdino.models import build_model
from SAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from SAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from SAM.segment_anything.segment_anything.predictor import SamPredictor
from SAM.segment_anything.segment_anything.build_sam import sam_model_registry
from SAM.segment_anything.segment_anything.build_sam_hq import sam_hq_model_registry
from diffusers.utils import check_min_version

import re
import yaml
from omegaconf import OmegaConf
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from torchvision.utils import save_image

import time
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from utils.src_files.helper_functions.bn_fusion import fuse_bn_recursively
from utils.src_files.models import create_model
from utils.src_files.models.tresnet.tresnet import InplacABN_to_ABN
import random

from transformers import AutoTokenizer, AutoModelForImageSegmentation
from utils.randeng.modeling_deltalm import DeltalmForConditionalGeneration
from torchvision import transforms
import webdataset
from torchdata.datapipes.iter import FileOpener
from torchdata.dataloader2 import DataLoader2

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

import webdataset as wds
import braceexpand
# from liqe import LIQE
import pyiqa

from torchvision.transforms import Normalize,ToTensor
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("Combined Inpainting Pipeline", add_help=True)
parser.add_argument("--ram_pretrained", type=str, default='gendata/RAM/pretrained/recognize-anything-plus-model/ram_plus_swin_large_14m.pth', help="RAM pretrained model path")
parser.add_argument("--ram_image_size", type=int, default=384, help="RAM image size")
parser.add_argument("--grounding_dino_config", type=str, default='gendata/SAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', help="Grounding DINO config file path")
parser.add_argument("--grounding_dino_checkpoint", type=str, default='gendata/SAM/groundingdino_swint_ogc.pth', help="Grounding DINO checkpoint file path")
parser.add_argument("--sam_version", type=str, default="vit_h", help="SAM ViT version: vit_b / vit_l / vit_h")
parser.add_argument("--sam_checkpoint", type=str, default='gendata/SAM/sam_vit_h_4b8939.pth', help="SAM checkpoint file path")
parser.add_argument('--num-classes', default=4088, type=int)
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)

######################################################################################################################################

parser.add_argument("--output_dir", type=str, default="outputs_flux_1")
parser.add_argument("--process_id", type=int, default=0)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda"
num_processors = 40


urls_ori = ["/mnt/data/group/text2img_data/kolors/*/*"]
# urls_ori = ["/mnt/data/group/text2img_data/flux_taisu/*/*"]


#rendeng
######################################################################################################################################
translation_model = DeltalmForConditionalGeneration.from_pretrained("/mnt/data/group/models/Randeng-Deltalm-362M-Zh-En", torch_dtype=torch.float16).to(device)
randeng_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/group/models/infoxlm-base")

def translate(text, device):
    translation_inputs = randeng_tokenizer(text, max_length=77, truncation=True, padding=True, return_tensors="pt")
    generate_ids = translation_model.generate(translation_inputs["input_ids"].to(device), max_new_tokens=16)
    en_text = randeng_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return en_text

#######################################################################################################################
bg_model = AutoModelForImageSegmentation.from_pretrained('/mnt/data/group/models/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][1])
bg_model.to(device)
bg_model.eval()
#######################################################################################################################


# RAM 模型加载
#########################################################################
ram_model = create_model(args, load_head=False).to(device)
state = torch.load(args.model_path, map_location='cpu')
ram_model.load_state_dict(state, strict=True)
ram_model = ram_model.cpu()
ram_model = InplacABN_to_ABN(ram_model)
ram_model = fuse_bn_recursively(ram_model)
ram_model = ram_model.to(device).half().eval()

with open("gendata/label_system_1015.txt", 'r') as f:
    classes_list = f.readlines()
    classes_list = [cla.strip() for cla in classes_list]

with open("gendata/del_labels_1224.txt", "r") as f:
    del_classes = f.readlines()
    del_classes = [cla.strip() for cla in del_classes]

def get_prompt_from_new_model(image):
    im = image
    im_resize = im.resize((args.image_size, args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).to(device).half()  # float16 inference
    output = torch.squeeze(torch.sigmoid(ram_model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    valid_indices = np.where(np_output > args.th)[0]
    detected_classes = [classes_list[i] for i in valid_indices]
    probability = [np_output[i] for i in valid_indices]
    output_classes = [cla for cla in detected_classes if (cla not in del_classes and "衣" not in cla)]
    return output_classes if output_classes else "None"
##############################################################################################################

############################################################################################################
def load_grounding_dino_model(config_path, checkpoint_path, device):
    dino_args = SLConfig.fromfile(config_path)
    dino_args.device = device
    model = build_model(dino_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

dino_model = load_grounding_dino_model(args.grounding_dino_config, args.grounding_dino_checkpoint, device=device)

sam_predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(device))

############################################################################################################

############################################################################################################
lama_model_path = "/mnt/data/group/models/big-lama/big-lama"
train_config_path = os.path.join(lama_model_path, 'config.yaml')

with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'
checkpoint_path = os.path.join(lama_model_path, 'models', 'best.ckpt')
lamamodel = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
lamamodel.freeze()
lamamodel.to(device)
############################################################################################################

############################################################################################################
model_ae, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
model_ae = model_ae.to(torch.bfloat16).to(device)
############################################################################################################

iqa_metric = pyiqa.create_metric("liqe_mix", pretrained=True,device=device)
iqa_metric_clip = pyiqa.create_metric("clipiqa+", pretrained=True,device=device) # qualiclip+ liqe_mix  clipiqa+  topiq_nr

def load_image(img):
    # load image
    image_pil = img # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
def collate_fn(examples):
    key = [example["__key__"] for example in examples]
    jpg = [example[list(example.keys())[1]] for example in examples ]
    
    json_data = [example[".json"] for example in examples]
    return { "jpg": jpg, "json":json_data, "key": key}

def decode(item):
    key, value = item

    if key.endswith(".jpg"):
        try:
            # Ensure we have bytes-like object
            if isinstance(value, bytes):
                from io import BytesIO
                value = Image.open(BytesIO(value)).convert("RGB")
            # If using torchdata's FileOpener, we might get a file-object
            else:
                value = Image.open(value).convert("RGB")
        except Exception as e:
            print(f"Reading {key} error, skip. Error: {e}")
            value = None
        return key, value
    elif key.endswith(".json"):
        value = json.load(value)
        return key, value

    else:
        value = None
        return key, value

def load_image_mask(img, mask_img, kernel_size=12):
    img_np = np.array(img.convert('RGB'))
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    out_img = img_np.astype('float32') / 255
    # Load the mask image
    mask_img = mask_img.resize(img.size, Image.LANCZOS)
    mask_img = mask_img.convert('L')
    mask_img = np.array(mask_img)
    mask_img[mask_img > 30] = 255


    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_img_dilated = cv2.dilate(mask_img.astype('uint8'), kernel, iterations=2)
    mask_img_dilated = mask_img_dilated.astype('float32') / 255
    mask_img_dilated = mask_img_dilated[None, :]

    processed_mask_img = Image.fromarray((mask_img_dilated[0] * 255).astype('uint8'))

    out_img = pad_img_to_modulo(out_img, 8)
    mask_img_dilated = pad_img_to_modulo(mask_img_dilated, 8)

    batch = {}
    batch["image"] = torch.tensor(out_img).unsqueeze(0)
    batch["mask"] = torch.tensor(mask_img_dilated).unsqueeze(0)
    return batch

def lama_model(batch, model, device):
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch = model(batch)
        cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_mask = batch["mask"][0].squeeze(0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
            cur_mask = cur_mask[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = Image.fromarray(cur_res.astype(np.uint8))

        cur_mask = np.clip(cur_mask * 255, 0, 255).astype('uint8')
        cur_mask = Image.fromarray(cur_mask.astype(np.uint8),mode='L')
    return cur_res,cur_mask

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


ASPECT_RATIO_1024_BIN = {
    "1": [1024, 1024]
}
def br(img):

    # Data settings
    rate = img.size[0]/img.size[1]
    rate_new = min(ASPECT_RATIO_1024_BIN.keys(), key=lambda ratio: abs(float(ratio) - rate))
    image_size = ASPECT_RATIO_1024_BIN[rate_new]
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = img
    input_images = transform_image(image).unsqueeze(0).to(device)

    
    with torch.no_grad():
        preds = bg_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    return mask

def calculate_overlap(backgroud, mask_img):
    # Convert PIL images into numpy arrays if they are not already
    if isinstance(backgroud, Image.Image):
        backgroud = np.array(backgroud)
    if isinstance(mask_img, Image.Image):
        mask_img = np.array(mask_img)
    if isinstance(backgroud, torch.Tensor):
        backgroud = backgroud.cpu().numpy()
    if isinstance(mask_img, torch.Tensor):
        mask_img = mask_img.cpu().numpy()
    # 二值化
    backgroud = backgroud > 0
    mask_img = mask_img > 0

    
    overlap_area = np.sum(backgroud & mask_img)#按位与
    mask_img_area = np.sum(mask_img)

    return overlap_area, mask_img_area
##############################################################################################################################
batch_size = args.batch_size

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def expand_urls1(urls):
    result = []
    for file_ in glob.glob(urls):
        result.append(file_)
    return result

all_urls = []
for url in urls_ori:
    if "*" in url:
        all_urls += expand_urls1(url)
    elif ".." in url:
        all_urls += expand_urls(url)
    else:
        all_urls = urls
print(len(all_urls))
all_urls = sorted(all_urls)

tar_list_split = np.array_split(all_urls, num_processors)[args.process_id]
print(f"nums of tars: {len(tar_list_split)}")
output_dir = f"{args.output_dir}/{args.process_id}"  ### need change
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

index = 0
num_per_dir = 5000
pattern = os.path.join(output_dir, f"%05d.tar")
sink = wds.ShardWriter(pattern, maxsize=int(6e10), maxcount=int(num_per_dir))


# Process all tar files in the directory
for tar_index,tar_name in enumerate(tar_list_split):
    dataset = FileOpener([tar_name], mode="b").load_from_tar().map(decode).webdataset().batch(batch_size).collate(collate_fn=collate_fn)
    dl = DataLoader2(dataset)
    try:

        for image_index, obj in enumerate(dl):
            img = obj["jpg"][0]
            jsonfile = obj["json"][0]

            # q_score_ori = liqe(ToTensor()(img).unsqueeze(0))
            q_score_ori = iqa_metric(ToTensor()(img).unsqueeze(0).to(device))[0]
            q_score_clip_ori = iqa_metric_clip(ToTensor()(img).unsqueeze(0).to(device))[0]
            if q_score_ori<3:
                continue

            if "aesthetic_score" in jsonfile:
                score_ori = jsonfile['aesthetic_score']
            else:
                score_ori = 0
            ori_imag = img
            try:
                prompt = get_prompt_from_new_model(img)
            except Exception as e:
                print('Error')
                continue
            if len(prompt) > 15:
                continue
            for prompt_i, zh_prompt in enumerate(prompt):
                en_prompt = translate([zh_prompt], device)[0]
                # Grounding DINO 
                image_pil, image_tensor = load_image(img)
                boxes_filt, pred_phrases = get_grounding_output(
                    dino_model, image_tensor, en_prompt, box_threshold=0.3, text_threshold=0.25, device=device
                )

                # SAM 
                image_cv_rgb = np.array(img)
                sam_predictor.set_image(image_cv_rgb)

                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv_rgb.shape[:2]).to(device)
                try:
                    masks, _, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False,
                    )
                except Exception as e:
                    # The problem with this is that there is no boundingbox
                    continue

                
                mask_img = torch.zeros(masks.shape[-2:])
                for idx, mask in enumerate(masks):
                    mask_img[mask.cpu().numpy()[0] == True] = 1
                
                ## Constraints on mask size
                combined_mask_np = mask_img.cpu().numpy()
                mask_ratio = np.sum(combined_mask_np) / combined_mask_np.size
                if mask_ratio < 0.02 or mask_ratio > 0.35:
                    continue
                
                ## Does the mask area meet the requirements in the foreground
                mask_br = br(img)
                mask_br_tensor = transforms.ToTensor()(mask_br)
                mask1_bool = mask_img.bool()
                mask2_bool = mask_br_tensor.bool()
                num_elements_mask1 = mask1_bool.sum().item()
                num_elements_in_mask2 = (mask1_bool & mask2_bool).sum().item()
                if not num_elements_in_mask2 / num_elements_mask1 >= 0.8:
                    continue
        
                combined_mask_pil = Image.fromarray((combined_mask_np * 255).astype(np.uint8))
        
                batch = load_image_mask(img, combined_mask_pil)
                try:
                    lama_output,mask = lama_model(batch, lamamodel, device)
                except Exception as e:
                    continue
                lama_output_tensor = preprocessor(images=lama_output, return_tensors="pt").pixel_values.to(torch.bfloat16).to(device)
                aesthetic_lama_output = model_ae(lama_output_tensor).logits.squeeze().float().cpu().detach().numpy()
                if aesthetic_lama_output < 5:
                    continue

                q_score = iqa_metric(ToTensor()(lama_output).unsqueeze(0).to(device))[0]
                q_score_clip = iqa_metric_clip(ToTensor()(lama_output).unsqueeze(0).to(device))[0]


                ## 25.03.26 Filter data with significant changes in absolute difference
                if q_score<3.5 or q_score_ori-q_score>0.2 or score_ori-aesthetic_lama_output>0.2 or q_score_clip_ori-q_score_clip>0.028:
                    # print(zh_prompt)
                    continue

                print(f"{tar_index}_{image_index}_{prompt_i} processing completed.")

                xkey = "%09d" % index
                sample = {
                        "__key__": xkey, 
                        "1.jpg": ori_imag,
                        "2.jpg": lama_output,
                        "3.jpg": mask,
                        "txt": zh_prompt,
                        "json": {
                            "caption_zh": jsonfile['caption_zh'],
                            "caption_en": jsonfile['caption_en'],
                            "instruction_en": f"Delete {en_prompt} in the picture",
                            "instruction_zh": f"把这张图片的{zh_prompt}删掉",
                            "label": "Delete",
                            "aesthetic_score_ori": float(score_ori),
                            "aesthetic_score_delete": float(aesthetic_lama_output),
                            "liqe_score_delete": float(q_score),
                            "liqe_score_ori": float(q_score_ori),
                            "liqe_score_clip_delete": float(q_score_clip),
                            "liqe_score_clip_ori": float(q_score_clip_ori)
                        },
                    }
                sink.write(sample)
                index += 1
                break
    except Exception as e:
        print(f"Error: {e}")
sink.close()


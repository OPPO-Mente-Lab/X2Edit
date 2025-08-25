
import json
import braceexpand
import webdataset as wds
from tqdm import tqdm
import torch
from torchvision.transforms.functional import crop
import re
from torchvision import transforms
import random
import numpy as np
import os,glob
from transformers import CLIPTextModel, T5TokenizerFast, CLIPTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,MT5EncoderModel,AutoTokenizer,AutoModel,AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor

from torchdata.datapipes.iter import FileLister, FileOpener
from torchdata.datapipes.iter import IterableWrapper

from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import random_split
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
from src.custom_multiplexer import SampleMultiplexer
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


class InverseNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 反向标准化: (tensor * std) + mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def extract_sentences(text):
    # 正则表达式匹配由大写字母组成的单词起始的句子
    pattern = r'[A-Z][A-Z-0-9]*.*?(?=\.\s|$)'
    matches = re.findall(pattern, text)

    return matches

def split_bucket(x):
    return x["bucket_id"]

class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+", default=["/mnt/workspace/group/text2img_data/X2I/tars/anyedit/*/*"])
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',default=True, action="store_true")
        parser.add_argument('--resample_train',default=True, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer_qwen,
        custom_collate_fn=None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer_qwen = tokenizer_qwen
        self.collate_fn = custom_collate_fn if custom_collate_fn is not None else collate_fn
        self.center_crop = args.center_crop
        self.resolution = args.resolution

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None


        self.prepare_data()
        self.setup()

    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            if "*" in url:
                all_urls += expand_urls1(url)
            else:
                all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)

    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer_qwen,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )

            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)

    # def _train_dataloader(self):

    #     if self.use_worker_init_fn:
    #         init_fn = worker_init_fn
    #     else:
    #         init_fn = None
    #     # return DataLoader(
    #     # num_workers=self.num_workers,
    #     return DataLoader(
    #         dataset=self.datasets['train'],
    #         num_workers=self.num_workers,
    #         batch_size=self.batch_size,
    #         prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
    #         pin_memory=True,
    #         shuffle=False,
    #         worker_init_fn=init_fn,
    #         collate_fn=self.collate_fn,
    #     )


    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        pipes_to_weights_dict = {}

        dp_list = IterableWrapper(self.datasets['train']).mydemux(
            num_instances=len(BUCKET_PROBS), classifier_fn=split_bucket, buffer_size=1000)

        for i in range(len(dp_list)):
            pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
        sample_mul_dp = SampleMultiplexer(
            pipes_to_weights_dict=pipes_to_weights_dict, batch_size=self.batch_size, seed=0).collate(collate_fn=collate_fn)
        mp_rs = MultiProcessingReadingService(num_workers=self.num_workers)
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
        return DataLoader2(sample_mul_dp, reading_service=rs)
        return DataLoaderX(
            self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

TETX_ENCODER = "chinese_clip"  ## mul_clip  chinese_clip  mt5  alt_clip

USED_KEYS = ["json"]
# BUCKETS = [[576, 1792], [640, 1600], [704, 1408], [768, 1280], [832, 1152], [896, 1088], [960, 1024], [1024, 1024], \
#     [1024, 960], [1088, 896], [1152, 832], [1280, 768], [1408, 704], [1600, 640], [1792, 576]]
# BUCKET_PROBS = [0.0010173, 0.000654022, 0.00305210, 0.11860329917, 0.088438340, 0.0194753, 0.009446987, 0.243695952, \
#     0.0123537, 0.02855897, 0.38478308, 0.051260809, 0.0302303611, 0.0061405421, 0.00228907]

BUCKETS = [[512, 512]]
BUCKET_PROBS = [1]

MAX_AR_ERROR = 2
ASPECTS = np.array([b[0]/b[1] for b in BUCKETS])

# all_dict = {
#     'controlnet':['controlnet','visual_sketch','visual_bbox'],
#     'removal': ['removal', 'ObjectRemoval', 'WaterMarkRemoval', 'SnowRemoval'],
#     'addition': ['addition', 'add', 'add object'],
#     'modification': ['attribute_modification', 'change material', 'material_change','color_alter','change arrangement', 'change color',  'change object', 'change expression', 'change direction', 'swap','replace', 'replace object'],
#     'enhancement': ['Dehazy', 'LowLightEnhance', 'Deblur'],
#     'style':['change style','style','style_change'],
#     'background':['background_change','change background',],
#     'transformation': [ 'appearance_alter', 'tune_transfer'],
#     'others': ['textual', 'outpainting', 'resize',  'relation', 'Colorization']
# }
# anyedit
# local_editing = ["remove", "replace", "add", "color_alter", "appearance_alter", "material_change", "action_change", "textual_change", "counting"]
# global_editing = ["background_change", "tone_transfer", "style_change"]
# camera_movement_editing = ["movement", "outpaint", "rotation_change", "resize"]
# implicit_editing = ["implicit_change", "relation_change"]
# visual_editing = ["visual_sketch", "visual_scribble", "visual_segmentation", "visual_depth", "visual_layout", "material_transfer", "image_reference"]


def to_type_id(target_value):
    all_dict = {
        '1': ['删除主体','removal','remove', 'ObjectRemoval', 'WaterMarkRemoval', 'SnowRemoval'],
        '2': ['添加主体','addition', 'add'],
        '3': ['风格更改','style','style_change','change style'],
        '4': ['背景更改','background_change','change background'],
        '5':['颜色更改','color_alter','attribute_modification','Colorization'],
		'6':['材料更改','material_alter','材质更改','material_change','visual_material_change'],
		'7':['动作更改','motion_change','action_change'],
		'8':['替换主体','subject-replace','swap','replace','replace object'],
		'9':['人物修图','ps_human','appearance_alter','change expression'],
		'10':['文字更改','text_change','text change'],
		'11':['色调变换','色调迁移','tune_transfer','tone_transfer'],
        '12':['相机移动','movement','rotation_change','resize','Camera Move Editing'],
        '13':['Reasoning','implicit_change','relation','Reasoning Editing','复杂推理'],
        '14':['low_leval'],
        '15':['subject','style transfer'],
    }
    for key, value_list in all_dict.items():
        if target_value in value_list:
            return int(key)
    return 0 


def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False

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


def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """

    for sample in samples:

        try:
            sample_json = sample["json"]
            if "1.0.jpg" in sample:
                w, h = sample["1.0.jpg"].size
            elif ".1.0.jpg" in sample:
                w, h = sample[".1.0.jpg"].size
                sample["1.0.jpg"] = sample.pop(".1.0.jpg")
                sample["2.jpg"] = sample.pop(".2.jpg")
            else:
                w, h = sample[".1.jpg"].size
                sample["1.0.jpg"] = sample.pop(".1.jpg")
                sample["2.jpg"] = sample.pop(".2.jpg")
        except:
            print("#######sample",sample)
            continue
        
        if w/h>1.3 or h/w>1.3:continue

        if "label" in sample_json:
            label = sample_json["label"]
            if label=="grit-entity-new": ## person
                if "caption_en_ori" not in sample_json and "caption_en" not in sample_json:
                    continue

            elif label=="AnyEdit": 
                if "label_fine" not in sample_json:
                    continue
            elif label=="Delete": 
                score_7b = sum(eval(sample_json["score"]))
                if score_7b<9:
                    continue
            elif label=="step1x": 
                if "score_7b" in sample_json:
                    score_7b = sum(eval(sample_json["score_7b"]))
                    liqe_score = sample_json["liqe_score"]
                    liqe_score_edit = sample_json["liqe_score_edit"]
                    liqe_score_clip = sample_json["liqe_score_clip"]
                    liqe_score_clip_edit = sample_json["liqe_score_clip_edit"]
                    aesthetic_score_v2_5 = sample_json["aesthetic_score_v2_5"]
                    aesthetic_score_v2_5_edit = sample_json["aesthetic_score_v2_5_edit"]
                    # if liqe_score_edit<3 and aesthetic_score_v2_5_edit<5 and liqe_score_clip_edit<0.65:
                    if liqe_score_edit<3 or aesthetic_score_v2_5_edit<4.8 or liqe_score_clip_edit<0.6:
                        continue
                    if score_7b<9:
                        continue
                elif "score" in sample_json and sample_json["score"]!="None":
                    score_72b = sum(eval(sample_json["score"]))
                    if score_72b<18:
                        continue

            elif label=="gpt4o": 
                if "score_7b" in sample_json:
                    if sample_json["task"] in ["动作更改","人物修图","文字更改"]:
                        continue


        if "caption_en" in sample_json and len(sample_json["caption_en"])==0: 
            continue


        aspect = float(w)/float(h)
        bucket_id = np.abs(ASPECTS - aspect).argmin()
        if abs(ASPECTS[bucket_id] - aspect) < MAX_AR_ERROR:
            sample["bucket_id"] = bucket_id
            yield sample


key_verifier = wds.filters.pipelinefilter(verify_keys)


def crop_left_upper(image, size):
    w, h = image.size

    detla_w = w-size[0]
    detla_h = h-size[1]
    x = random.randint(0, detla_w)
    y = random.randint(0, detla_h)
    return (y, x), crop(image, y, x, size[1], size[0])


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer_qwen = None,
            hr_size=-1,
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=True,
            shuffle_shards=True,
            center_crop=False
    ):

        super().__init__()
        keys = USED_KEYS
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.center_crop = center_crop
        # self.crop = transforms.CenterCrop(size) if center_crop else crop_left_upper
        # self.crop = transforms.CenterCrop(size)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer_qwen = tokenizer_qwen
        self.append(wds.ResampledShards(urls))
        # self.append(wds.SimpleShardList(urls))
        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

        self.tokenizer_max_length = 1024
        # self.prompt_template_encode = '''<|im_start|>system\nDescribe the image by detailing the color, shape, size, 
        #         texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'''
        self.prompt_template_encode = '''
                <|im_start|>system
                Describe the key features of the input image (color, shape, size, texture, objects, background), then
                explain how the user’s text instruction should alter or modify the image. Generate a new image that
                meets the user’s requirements while maintaining consistency with the original input where appropriate.
                <|im_end|>
                <|im_start|>user
                <|vision_start|><|user_image|><|vision_end|>{}<|im_end|>
                <|im_start|>assistant
                '''
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

    def preproc(self, sample):
        """Applies the preprocessing for images"""
        pattern = r'<\|([^|]+)\|>'
        example = {}
        sample_json = sample["json"]
        example["bucket_id"] = sample["bucket_id"]
        # resize
        dst_size = BUCKETS[sample["bucket_id"]]
        if dst_size==[512, 512]:
            proportion=1
        else:
            proportion=2
        if "label" in sample_json:
            label = sample_json["label"]
            if label=="Subjects200K" and "1.1.jpg" in sample:  ## canny depth
                if random.random()>0.5:
                    instruction_en = "Canny image generates images"
                    instance_image_infer = sample[f"1.1.jpg"].convert("RGB")
                else:
                    instruction_en = "Depth image generates images"
                    instance_image_infer = sample[f"1.2.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = "low_leval"
                

            elif label=="Subjects200K":
                instruction_en = sample_json["caption_en"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = "subject"

            # elif label=="grit-entity-new": ## person
            #     if "caption_en_ori" in sample_json:
            #         instruction_en = sample_json["caption_en_ori"]
            #     else:
            #         instruction_en = sample_json["caption_en"]
            #     instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            #     instance_image = sample[f"2.jpg"].convert("RGB")

            # elif label=="gopro":## 多条件
            #     instruction_en = sample_json["caption_en"]
            #     instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            #     instance_image = sample[f"2.jpg"].convert("RGB")

            # elif label=="derain":## 多条件
            #     instruction_en = sample_json["caption_en"]
            #     instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            #     instance_image = sample[f"2.jpg"].convert("RGB")

            # elif label=="enhance":## 多条件
            #     instruction_en = sample_json["caption_en"]
            #     instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            #     instance_image = sample[f"2.jpg"].convert("RGB")

            elif label=="AnyEdit": 
                instruction_en = sample_json["caption_en"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["label_fine"]
                # print(edit_type)
                # if edit_type in [
                #         "visual sketch",
                #         "visual scribble",
                #         "visual segmentation",
                #         "visual depth",
                #         "visual layout",
                #         "material transfer",
                #         "image reference"
                #     ]:
                #     print(edit_type)

            elif label=="Style Transfer": 
                instruction_en = sample_json["style_zh"]
                instance_image = sample[f"png"].convert("RGB")
                instance_image_infer = sample[f"jpg"].convert("RGB")
                edit_type = 'style'
                
            elif label=="Delete": ## 自构造
                if int(sample["__key__"])%2:
                    instruction_en = sample_json["instruction_zh"]
                else:
                    instruction_en = sample_json["instruction_en"]
                instance_image = sample[f"2.jpg"].convert("RGB")
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                edit_type = 'removal'
                # else:
                #     instruction_en = sample_json["instruction_en"].replace("Delete","Add")
                #     instance_image = sample[f"1.jpg"].convert("RGB")   
                #     instance_image_infer = sample[f"2.jpg"].convert("RGB")
                #     edit_type = 'add'
            
            elif label=="PromptfixData": 
                instruction_en = sample_json["caption_en"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["label_fine"]

            # elif label=="SEED-Data": 
            #     instruction_en = sample_json["caption_en"]
            #     instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            #     instance_image = sample[f"2.jpg"].convert("RGB")
            #     edit_type = sample_json["label_fine"]

            elif label=="Kontext": 
                if random.random()>0.5:
                    instruction_en = sample_json["instruction"]
                else:
                    instruction_en = sample_json["instruction_zh"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["task"]

            elif label=="step1x": 
                instruction_en = sample_json["instruction"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["task"]

            elif label=="gpt4o": 
                if random.random()>0.5:
                    instruction_en = sample_json["instruction"]
                else:
                    instruction_en = sample_json["instruction_zh"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["task"]

            elif label=="OmniConsistency": 
                instruction_en = sample_json["instruction"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["task"]
                if edit_type!="text change":
                    edit_type="style"

            else:
                instruction_en = sample_json["instruction"]
                instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
                instance_image = sample[f"2.jpg"].convert("RGB")
                edit_type = sample_json["task"]

        else: ## stepX构造数据
            instruction_en = sample_json["instruction"]
            instance_image_infer = sample[f"1.0.jpg"].convert("RGB")
            instance_image = sample[f"2.jpg"].convert("RGB")
            edit_type = sample_json["task"]

        # instruction_en += ","+edit_type
        # instance_image_infer.save("1.png")
        # instance_image.save("2.png")
        # print(instruction_en)
        example["task"] = to_type_id(edit_type)
        example["original_size"] = instance_image_infer.size
        if int(example["original_size"][0]*dst_size[1]/example["original_size"][1]) >= dst_size[0]:
            instance_image_infer = transforms.Resize((int(dst_size[1]/proportion), int(
                example["original_size"][0]*dst_size[1]/example["original_size"][1]/proportion)), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image_infer)
        else:
            instance_image_infer = transforms.Resize((int(example["original_size"][1]*dst_size[0]/example["original_size"][0]/proportion),
                                            int(dst_size[0]/proportion)), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image_infer)

        example["original_size"] = instance_image.size
        if int(example["original_size"][0]*dst_size[1]/example["original_size"][1]) >= dst_size[0]:
            instance_image = transforms.Resize((dst_size[1], int(
                example["original_size"][0]*dst_size[1]/example["original_size"][1])), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
        else:
            instance_image = transforms.Resize((int(example["original_size"][1]*dst_size[0]/example["original_size"][0]),
                                            dst_size[0]), interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)

        instance_image = transforms.CenterCrop(dst_size[::-1])(instance_image)
        instance_image_infer = transforms.CenterCrop(dst_size[::-1])(instance_image_infer)
        example["instance_image_output"] = self.image_transforms(instance_image)
        example["instance_image_input"] = self.image_transforms(instance_image_infer)

        # instance_image_infer.save(f"tmp/{instruction_en[:100]}_in.jpg")
        # instance_image.save(f"tmp/{instruction_en[:100]}_out.jpg")

        # instance_image_infer_qwen = instance_image_infer.resize((224,224))
        # contents=[{"type": "text", "text": instruction_en},{"type": "image", "image": instance_image_infer_qwen}]
        # prompt = [instruction_en]
        # template = self.prompt_template_encode
        # txt = [template.format(e) for e in prompt]

        # self.prompt_template_encode = '''<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then 
        # explain how the user's text instruction should alter or modify the image. Generate a new image that 
        # meets the user's requirements while maintaining consistency with the original input where appropriate.
        # <|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n'''


        drop_idx = self.prompt_template_encode_start_idx
        txt = [
            f'''
                <|im_start|>system
                Describe the key features of the input image (color, shape, size, texture, objects, background), then
                explain how the user’s text instruction should alter or modify the image. Generate a new image that
                meets the user’s requirements while maintaining consistency with the original input where appropriate.
                <|im_end|>
                <|im_start|>user
                <|vision_start|><{instance_image_infer}><|vision_end|>{instruction_en}<|im_end|>
                <|im_start|>assistant
                '''
        ]
        # txt_tokens = self.tokenizer_qwen(txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt")
        txt_tokens = self.tokenizer_qwen(txt, max_length=self.tokenizer_max_length + drop_idx, padding="max_length", truncation=True, return_tensors="pt")

        example["input_ids"] = txt_tokens.input_ids
        example["attention_mask"] = txt_tokens.attention_mask

        return example


def collate_fn(examples):
    input_ids = [example["input_ids"] for example in examples]

    pixel_values_output = [example["instance_image_output"] for example in examples]
    pixel_values_input = [example["instance_image_input"] for example in examples]

    # pixel_values = [example["pixel_values"] for example in examples]
    # image_grid_thw = [example["image_grid_thw"] for example in examples]

    attention_mask = [example["attention_mask"] for example in examples]
    task = [torch.tensor(example["task"]) for example in examples]

    batch = {
        "input_ids": torch.cat(input_ids),
        "attention_mask": torch.cat(attention_mask),
        # "pixel_values":  torch.stack(pixel_values),
        # "image_grid_thw": torch.cat(image_grid_thw),
        "pixel_values_output": torch.stack(pixel_values_output),
        "pixel_values_input": torch.stack(pixel_values_input),
        "task": torch.stack(task)

    }

    return batch


if __name__ == '__main__':
    inverse_normalize = InverseNormalize(mean=[0.5], std=[0.5])
    device = "cuda"
    weight_dtype = torch.bfloat16
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/web-image/*/*"]
    urls=["/mnt/workspace/group/text2img_data/X2I/tars/subject/Subjects200K/1/*"]
    urls=["/mnt/workspace/group/text2img_data/X2I/tars/subject/Subjects200K/0/*"]
    # urls=["/mnt/workspace/group/xuguo/gendata/outputs_flux/*/*"]
    # # urls=["/mnt/workspace/group/text2img_data/style/*/*"]
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/anyedit/*/*"]
    # # # # urls=["/mnt/workspace/group/text2img_data/X2I/tars/OmniEdit/*/*"]
    # # # urls=["/mnt/workspace/group/text2img_data/X2I/tars/PromptfixData/*/*"]
    # # # urls=["/mnt/workspace/group/text2img_data/X2I/tars/SEED-Data/*/*"]
    urls=["/mnt/workspace/group/text2img_data/X2I/tars/style//*/*"]
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/step1x_7b_72b/*/*"]
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/bagel_stage1/*/*"]
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/delete/*/*"]
    urls=["/mnt/workspace/group/text2img_data/X2I/tars/Kontext/*/*"]
    urls=["/mnt/workspace/group/******/textflux-main/textflux-qwen-generate-ch/*/*"]
    # urls=["/mnt/workspace/group/text2img_data/X2I/tars/Kontext_aspect/*/*"]


    all_urls = []
    for url in urls:
        if "*" in url:
            all_urls += expand_urls1(url)
        elif ".." in url:
            all_urls += expand_urls(url)
        else:
            all_urls = urls
    print(len(all_urls))
    model_name = "/mnt/workspace/group/models/Qwen-Image/tokenizer/"
    tokenizer_qwen = Qwen2Tokenizer.from_pretrained(model_name)
    ds = ImageEmbeddingDataset(
        all_urls,
        tokenizer_qwen,
        resample=True,
        hr_size=512,
        handler=wds.handlers.warn_and_continue
    )

    source_dp = IterableWrapper(ds)

    def split_bucket(n):
        return n["bucket_id"]

    dp_list = source_dp.mydemux(num_instances=len(BUCKETS), classifier_fn=split_bucket)
    pipes_to_weights_dict = {}

    for i in range(len(dp_list)):
        pipes_to_weights_dict[dp_list[i]] = BUCKET_PROBS[i]
    sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=pipes_to_weights_dict, batch_size=1, seed=0).collate(collate_fn=collate_fn)
    mp_rs = MultiProcessingReadingService(num_workers=1)
    dl = DataLoader2(sample_mul_dp, reading_service=mp_rs)

    # dl = DataLoader(
    #         ds,
    #         num_workers=1,
    #         batch_size=1,
    #         prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
    #         pin_memory=True,
    #         shuffle=False,
    #         collate_fn=collate_fn
    #     )

    for i, batch in enumerate(tqdm(dl)):
        if i<100:
            save_image(inverse_normalize(batch["pixel_values_input"]),f"tmp/{i}_in.jpg")
            save_image(inverse_normalize(batch["pixel_values_output"]),f"tmp/{i}_out.jpg")
        # if i%1000==0:
        #     print(i,batch["task"])

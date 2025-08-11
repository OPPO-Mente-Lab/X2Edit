import logging
import os
from urllib import request

import torch
import torch.nn as nn
from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)
from ..tresnet import TResnetM, TResnetL, TResnetXL
from transformers import CLIPModel


def create_model(args,load_head=False, train_encoder=True):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()
    
    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    ####################################################################################
    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth") and model_path == "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth":
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu') ## OrderedDict对象
        if 'model' in state:
            key = 'model'
        else:
            key = 'state_dict'
        if not load_head:
            if model_path.endswith(".ckpt"):
                filtered_dict = {k: v for k,v in state.items() if 'head.decoder' not in k}
                model.load_state_dict(filtered_dict, strict=False)
            else:
                filtered_dict = {k: v for k, v in state[key].items() if
                                (k in model.state_dict() and 'head.decoder' not in k)}
                model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state, strict=True)
    
    if not train_encoder:
        for name, param in model.named_parameters():
            if "body" in name:
                param.requires_grad = False

    return model


class MyModelWithClassifier(nn.Module):
    def __init__(self, args):
        vit_path = '/mnt/data/group/models/clip-vit-large-patch14'
        super(MyModelWithClassifier, self).__init__()
        clip = CLIPModel.from_pretrained(vit_path)
        self.model = add_ml_decoder_head(clip.vision_model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                decoder_embedding=args.decoder_embedding, zsl=args.zsl)

    def forward(self, images):
        features = self.model(images).last_hidden_state
        if hasattr(self.model, 'head'):  # 检查是否存在分类头
            outputs = self.model.head(features)
        else:
            outputs = features  # 如果没有分类头，直接返回特征（可根据实际情况调整）
        return outputs

def create_model_vit(args,load_head=False, train_encoder=True):
    """Create a model
    """
    model = MyModelWithClassifier(args)
    if not train_encoder:
        for name, param in model.named_parameters():
            print(name)
            if "decoder" not in name:
                param.requires_grad = False
    return model
import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from src.mobilenetv4_blocks import UniversalInvertedResidual

class MLP3(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, layer_norm_eps=1e-5, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(out_dim, out_dim1)
        )

    def forward(self, x):
        x = self.layernorm(x)
        x2 = self.projector(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2

class Proj7Exp(nn.Module):
    def __init__(self, in_channels=25, kernel_size=5, input_dim=896, output_dim0=768, output_dim1=4096, num_layers=2, num_heads=12, norm_eps=1e-6, head_dim=64, use_t5=True, use_scale=True, use_cnn=True) -> None:
        super().__init__()
        self.use_t5 = use_t5
        self.use_scale = use_scale
        self.use_cnn = use_cnn
        if self.use_t5:
            config = T5Config(num_heads=num_heads, num_layers=num_layers, num_decoder_layers=0, layer_norm_epsilon=norm_eps, is_encoder_decoder=False, 
                is_decoder=False, d_ff=input_dim*4, d_kv=head_dim, d_model=input_dim, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu", use_cache=False)
            print(f"config: {config}")

            self.t5stack = T5Stack(config)
        if self.use_scale:
            self.cha_scale = nn.Parameter(torch.empty(1, in_channels, 1, 1), requires_grad=True)
        elif self.use_cnn:
            self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.mlp = MLP3(input_dim, output_dim1, output_dim1, output_dim0, norm_eps)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.use_scale:
            nn.init.xavier_normal_(self.cha_scale, gain=1)

    def enable_gradient_checkpointing(self):
        self.t5stack.gradient_checkpointing_enable()
        
    def forward(self, x):
        B, C, S, H = x.shape
        if self.use_t5:
            x = self.t5stack(inputs_embeds=x.contiguous().view(B * C, S, H)).last_hidden_state
        if self.use_scale:
            x = (self.cha_scale * x.view(B, C, S, H)).mean(dim=1)
        elif self.use_cnn:
            x = self.conv(x.view(B, C, S, H)).squeeze(1)
        else:
            x = x.view(B, C, S, H).mean(dim=1)
        return self.mlp(x)

def create_proj3_qwen3b(in_channels, use_t5=True, use_scale=True, use_cnn=False):
    use_cnn = False if use_scale else use_cnn
    print(f"xxxxxx create Proj7Exp, in_channels: {in_channels}, use_t5: {use_t5}, use_scale: {use_scale}, use_cnn: {use_cnn}")
    return Proj7Exp(in_channels=in_channels, kernel_size=5, input_dim=2048, output_dim0=768, output_dim1=4096, num_layers=2, num_heads=28, norm_eps=1e-6, head_dim=128, use_t5=use_t5, use_scale=use_scale, use_cnn=use_cnn)

def create_proj3_qwen7b(in_channels, use_t5=True, use_scale=True, use_cnn=False):
    use_cnn = False if use_scale else use_cnn
    print(f"xxxxxx create Proj7Exp, in_channels: {in_channels}, use_t5: {use_t5}, use_scale: {use_scale}, use_cnn: {use_cnn}")
    return Proj7Exp(in_channels=in_channels, kernel_size=5, input_dim=3584, output_dim0=768, output_dim1=4096, num_layers=2, num_heads=28, norm_eps=1e-6, head_dim=128, use_t5=use_t5, use_scale=use_scale, use_cnn=use_cnn)

class Proj8(nn.Module):

    def __init__(self, proj, ref_in_channels=16, ref_out_channels=16 * 16, ref_kernel_size=5, ref_exp_ratio=4.0, ref_num_layers=8) -> None:
        super().__init__()
        self.proj = proj
        uir_args = {
            "in_chs": ref_in_channels, 
            "out_chs": ref_in_channels, 
            "dw_kernel_size_start": ref_kernel_size, 
            "dw_kernel_size_mid": ref_kernel_size, 
            "exp_ratio": ref_exp_ratio,
        }

        uir_list = [UniversalInvertedResidual(**uir_args) for i in range(ref_num_layers - 1)]
        uir_args["out_chs"] = ref_out_channels
        uir_list.append(UniversalInvertedResidual(**uir_args))
        self.uirs = nn.Sequential(*uir_list)
    
    ## B*77*4096 --> B*1280   B*77*2048  
    def forward(self, x, y):
        x0, x1 = self.proj(x)
        y = self.uirs(y)
        return x0, x1, y
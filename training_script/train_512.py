import argparse
import copy
import logging
import math
import os
import gc
import shutil
from pathlib import Path
import re
from tqdm.auto import tqdm
from PIL import Image

from safetensors.torch import save_file
from safetensors import safe_open

import accelerate
from accelerate.state import AcceleratorState
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import transformers
from transformers.utils import ContextManagers
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module

from src.proj import create_proj3_qwen7b
from src.prompt_helper import *
from src.pipeline import FluxPipeline, resize_position_encoding, prepare_latent_subject_ids
from src.jsonl_datasets import make_train_dataset, collate_fn
from src.custom_dataset_512 import DataModuleCustom 
from src.transformer_flux import FluxTransformer2DModel
from src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor

from torch import nn
import torch.nn.functional as F
from torch.distributed import all_gather, get_world_size, get_rank

logger = get_logger(__name__)


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/workspace/group/models/flux/FLUX.1-dev",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--tasks",
        type=int,
        default=16
    )

    parser.add_argument(
        "--experts",
        type=int,
        default=12
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/workspace/group/******/AndesDiT/EasyControl/train/models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )

    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--X2I_mllm",
        default="/mnt/workspace/group/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--X2I_alignet",
        default= "/mnt/workspace/group/**/flux/result_fit_speed/qwenvl25_dev_norm/57000/diffusion_pytorch_model.bin",
    )
    parser = DataModuleCustom.add_data_specific_args(parser)
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args



class disp_loss(torch.nn.Module):
    def __init__(self, transformer, proj_t5, task_embeddings_lora, use_Dispersive_Loss=False):
        super().__init__()
        self.transformer = transformer
        self.proj_t5 = proj_t5
        self.task_embeddings_lora = task_embeddings_lora
        self.use_Dispersive_Loss = use_Dispersive_Loss

    def compute_disp_loss(self, hidden_states_all, task_type=None, loss_weight=0.1, temperature=0.5):
        total_loss = 0.0
        num_hidden_states = len(hidden_states_all)
        for hidden_states in hidden_states_all:
            hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            # hidden_states_norm = torch.nn.functional.normalize(hidden_states, dim=-1) 
            hidden_states_norm = F.normalize(hidden_states, p=2, dim=1)

            # ## 实现1
            # sim_matrix = torch.mm(hidden_states_norm, hidden_states_norm.T)  # 余弦相似度
            # sim_matrix = sim_matrix / temperature
            # pos_mask = (task_type.unsqueeze(1) == task_type.unsqueeze(0)).float()
            # pos_mask.fill_diagonal_(0)
            # loss = -torch.log(
            #     (torch.sum(torch.exp(sim_matrix)*pos_mask, dim=1)+ 1e-8) / 
            #     (torch.sum(torch.exp(sim_matrix), dim=1) + 1e-8)
            # ).mean()

            ## 实现2
            dist_matrix = torch.cdist(hidden_states_norm, hidden_states_norm, p=2)** 2
            dist_matrix = dist_matrix / temperature
            pos_mask = (task_type.unsqueeze(1) == task_type.unsqueeze(0)).float()
            pos_mask.fill_diagonal_(0)
            # Calculate the loss using L2 distance (smaller distances are better)
            loss = -torch.log(
                (torch.sum(torch.exp(-dist_matrix) * pos_mask, dim=1) + 1e-8) /
                (torch.sum(torch.exp(-dist_matrix), dim=1) + 1e-8)
            ).mean()

            # # # 实现2
            # d_matrix = torch.cdist(hidden_states_norm, hidden_states_norm, p=2)** 2 
            # d_matrix = d_matrix/ temperature
            # # d_matrix_all = d_matrix[d_matrix > 0]
                    
            # pos_mask = (task_type.unsqueeze(1) == task_type.unsqueeze(0)).float()
            # neg_mask = 1 - pos_mask
            # pos_mask.fill_diagonal_(0)  # 排除自身
            # d_matrix_neg = d_matrix * neg_mask
            # d_matrix_pos = d_matrix * pos_mask
            # if len(d_matrix_neg) == 0:
            #     nage_sim = torch.tensor(0.0, device=hidden_states.device)
            # else:
            #     nage_sim = torch.log(torch.mean(torch.exp(-d_matrix_neg)))
            # pos_sim = torch.mean(torch.sum(d_matrix_pos , dim=1) / (pos_mask.sum(dim=1) + 1e-8))
            # loss = nage_sim + pos_sim


            # margin = 1
            # # 计算所有样本对的余弦相似度矩阵
            # sim_matrix = torch.mm(hidden_states_norm, hidden_states_norm.t())  # [batch, batch]
            # # 创建正负样本掩码
            # pos_mask = (task_type.unsqueeze(1) == task_type.unsqueeze(0)).float()
            # pos_mask.fill_diagonal_(0)
            # neg_mask = (task_type.unsqueeze(1) != task_type.unsqueeze(0)).float()
            # # 计算正样本和负样本的最大相似度
            # pos_sim = (sim_matrix * pos_mask).max(dim=1)[0]  # [batch]
            # neg_sim = (sim_matrix * neg_mask).max(dim=1)[0]  # [batch]
            # # 计算Hinge Loss
            # loss = torch.clamp(neg_sim - pos_sim + margin, min=0).mean()
            
            #  Covariance Loss
            # # 计算特征协方差矩阵
            # cov_matrix = torch.mm(hidden_states_norm, hidden_states_norm.t()) / hidden_states_norm.size(0)
            # # 创建任务类型掩码
            # task_mask = (task_type.unsqueeze(1) == task_type.unsqueeze(0)).float()
            # # 计算类内协方差（正样本）
            # intra_cov = (cov_matrix * task_mask).sum() / task_mask.sum()
            # # 计算类间协方差（负样本）
            # inter_cov = (cov_matrix * (1 - task_mask)).sum() / (1 - task_mask).sum()
            # # 计算协方差差异损失
            # loss = torch.abs(intra_cov - inter_cov)

            total_loss += loss

        return total_loss / num_hidden_states * loss_weight
    def compute_disp_loss4(self, hidden_states_all, task_type=None, loss_weight=0.2, temperature=0.5):
        total_loss = 0.0
        hidden_states = hidden_states_all[4].view(hidden_states_all[4].shape[0], -1)
        hidden_states_norm = F.normalize(hidden_states, p=2, dim=1)
        mbs = hidden_states_norm.shape[0]

        # 全局特征聚合
        world_size = get_world_size()
        rank = get_rank()
        gathered_states = [torch.zeros_like(hidden_states_norm) for _ in range(world_size)]
        all_gather(gathered_states, hidden_states_norm)
        global_states = torch.cat(gathered_states, dim=0)

        gathered_task_types = [torch.zeros_like(task_type) for _ in range(world_size)]
        all_gather(gathered_task_types, task_type)
        gathered_task_types = torch.cat(gathered_task_types, dim=0)

        # 计算分子
        dist_matrix = torch.cdist(hidden_states_norm, global_states, p=2)** 2
        dist_matrix = dist_matrix / temperature
        pos_mask = (task_type.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
        pos_mask[:, rank * mbs : (rank + 1) * mbs].fill_diagonal_(0)
        a = torch.sum(torch.exp(-dist_matrix) * pos_mask, dim=1)

        # 计算分母
        # dist_matrix = torch.cdist(global_states, global_states, p=2)** 2
        # dist_matrix = dist_matrix[rank * mbs : (rank + 1) * mbs]
        # dist_matrix = dist_matrix / temperature
        # pos_mask = (gathered_task_types.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
        # pos_mask.fill_diagonal_(0)
        b = torch.sum(torch.exp(-dist_matrix), dim=1)
        # Calculate the loss using L2 distance (smaller distances are better)
        loss = -torch.log(
            (a + 1e-8) /
            (b + 1e-8)
        ).mean()

        return loss * loss_weight


    def compute_disp_loss_all(self, hidden_states_all, task_type=None, loss_weight=0.2, temperature=0.5):
        total_loss = 0.0
        num_hidden_states = len(hidden_states_all)
                    
        world_size = get_world_size()
        rank = get_rank()
        
        gathered_task_types = [torch.zeros_like(task_type) for _ in range(world_size)]
        task_type_work = all_gather(gathered_task_types, task_type, async_op=True)
        
        global_states_work = None
        gathered_states = None
        hidden_states = hidden_states_all[0]
        mbs = hidden_states.shape[0]
        hidden_states_norm = F.normalize(hidden_states.view(mbs, -1), p=2, dim=-1)
        del hidden_states
        _, h = hidden_states_norm.shape
        dtype = hidden_states_norm.dtype
        device = hidden_states_norm.device

        # 全局特征聚合
        global_states = torch.zeros((mbs*world_size, h), dtype=dtype, device=device, requires_grad=False)
        gathered_states = list(torch.chunk(global_states, world_size, dim=0))
        global_states_work = all_gather(gathered_states, hidden_states_norm, async_op=True)
        del gathered_states
        

        with torch.no_grad():
            task_type_work.wait()
            gathered_task_types = torch.cat(gathered_task_types, dim=0)
            pos_mask = (task_type.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
            pos_mask[:, rank * mbs : (rank + 1) * mbs].fill_diagonal_(0)


        for i in range(num_hidden_states):
            if i + 1 < num_hidden_states:
                hidden_states_norm2 = F.normalize(hidden_states_all[i + 1].view(mbs, -1), p=2, dim=-1)
            global_states_work.wait()
            # 计算分子
            dist_matrix = torch.cdist(hidden_states_norm, global_states, p=2, compute_mode='use_mm_for_euclid_dist')** 2
            del global_states
            
            if i + 1 < num_hidden_states:
                # 全局特征聚合
                global_states = torch.zeros((mbs*world_size, h), dtype=dtype, device=device, requires_grad=False)
                gathered_states = list(torch.chunk(global_states, world_size, dim=0))
                global_states_work = all_gather(gathered_states, hidden_states_norm2, async_op=True)
                del gathered_states
            del hidden_states_norm
            dist_matrix = torch.exp(-(dist_matrix / temperature))
            a = torch.sum(dist_matrix * pos_mask, dim=1)
            # 计算分母
            b = torch.sum(dist_matrix, dim=1)
            del dist_matrix
            # Calculate the loss using L2 distance (smaller distances are better)
            loss = -torch.log(
                (a + 1e-8) /
                (b + 1e-8)
            ).mean()
            total_loss += loss

            hidden_states_norm = hidden_states_norm2
        del pos_mask
        return total_loss / num_hidden_states * loss_weight

    def compute_disp_loss_all2(self, hidden_states_all, task_type=None, loss_weight=0.2, temperature=0.5):
        total_loss = 0.0
        num_hidden_states = len(hidden_states_all)
                    
        world_size = get_world_size()
        rank = get_rank()
        
        gathered_task_types = [torch.zeros_like(task_type) for _ in range(world_size)]
        task_type_work = all_gather(gathered_task_types, task_type, async_op=True)

        mbs = hidden_states_all[0].shape[0]
        hidden_states = torch.stack([i.view(mbs, -1) for i in hidden_states_all], dim=0)
        hidden_states_norm = F.normalize(hidden_states, p=2, dim=2)
        L, B, H = hidden_states_norm.shape

        with torch.no_grad():
            task_type_work.wait()
            gathered_task_types = torch.cat(gathered_task_types, dim=0)
            pos_mask = (task_type.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
            pos_mask[:, rank * mbs : (rank + 1) * mbs].fill_diagonal_(0)

            global_states = torch.empty((L, B * world_size, H), dtype=hidden_states_norm.dtype, device=hidden_states_norm.device, requires_grad=False)
            gathered_states = list(torch.chunk(global_states, chunks=world_size, dim=1))
            all_gather(gathered_states, hidden_states_norm, async_op=False)
        
        for i in range(L):
            dist_matrix = torch.cdist(hidden_states_norm[i], global_states[i], p=2, compute_mode='use_mm_for_euclid_dist')** 2
            dist_matrix = torch.exp(-(dist_matrix / temperature))
            a = torch.sum(dist_matrix * pos_mask, dim=1)
            # 计算分母
            b = torch.sum(dist_matrix, dim=1)
            del dist_matrix
            # Calculate the loss using L2 distance (smaller distances are better)
            loss = -torch.log(
                (a + 1e-8) /
                (b + 1e-8)
            ).mean()
            total_loss += loss
        # del hidden_states
        # del hidden_states_norm
        # del pos_mask
        # del global_states
        # del gathered_states
        # gc.collect()

        return total_loss / L * loss_weight


    def compute_disp_loss_all_ori(self, hidden_states_all, task_type=None, loss_weight=0.2, temperature=0.5):
        total_loss = 0.0
        num_hidden_states = len(hidden_states_all)
        for hidden_states in hidden_states_all:
            hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            # hidden_states_norm = torch.nn.functional.normalize(hidden_states, dim=-1) 
            hidden_states_norm = F.normalize(hidden_states, p=2, dim=1)

            mbs = hidden_states_norm.shape[0]
            # 全局特征聚合
            world_size = get_world_size()
            rank = get_rank()
            gathered_states = [torch.zeros_like(hidden_states_norm) for _ in range(world_size)]
            all_gather(gathered_states, hidden_states_norm)
            global_states = torch.cat(gathered_states, dim=0)

            gathered_task_types = [torch.zeros_like(task_type) for _ in range(world_size)]
            all_gather(gathered_task_types, task_type)
            gathered_task_types = torch.cat(gathered_task_types, dim=0)

            # 计算分子
            dist_matrix = torch.cdist(hidden_states_norm, global_states, p=2, compute_mode='use_mm_for_euclid_dist')** 2
            dist_matrix = dist_matrix / temperature
            pos_mask = (task_type.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
            pos_mask[:, rank * mbs : (rank + 1) * mbs].fill_diagonal_(0)
            a = torch.sum(torch.exp(-dist_matrix) * pos_mask, dim=1)

            # 计算分母
            # dist_matrix = torch.cdist(global_states, global_states, p=2)** 2
            # dist_matrix = dist_matrix[rank * mbs : (rank + 1) * mbs]
            # dist_matrix = dist_matrix / temperature
            # pos_mask = (gathered_task_types.unsqueeze(1) == gathered_task_types.unsqueeze(0)).float()
            # pos_mask.fill_diagonal_(0)
            b = torch.sum(torch.exp(-dist_matrix), dim=1)
            # Calculate the loss using L2 distance (smaller distances are better)
            loss = -torch.log(
                (a + 1e-8) /
                (b + 1e-8)
            ).mean()
            total_loss += loss

        return total_loss / num_hidden_states * loss_weight


    def forward(self,task_type,packed_noisy_model_input,cond_packed_noisy_model_input,
                timesteps,guidance,text_embeddings,latent_image_ids):
        pooled_prompt_embeds, prompt_embeds = self.proj_t5(text_embeddings)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=pooled_prompt_embeds.dtype, device=pooled_prompt_embeds.device)
        task_type=task_type.to(dtype=torch.int)
        task_embeddings = self.task_embeddings_lora(task_type)      

        model_pred, hidden_states_all = self.transformer(
            hidden_states=packed_noisy_model_input,
            cond_hidden_states=cond_packed_noisy_model_input,
            task_embeddings = task_embeddings,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )

        if self.use_Dispersive_Loss:
            return model_pred, hidden_states_all
        else:
            return model_pred, None

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs()
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    # text_encoder_cls_one = import_model_class_from_model_name_or_path(
    #     args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    # )
    # text_encoder_cls_two = import_model_class_from_model_name_or_path(
    #     args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    # )
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder_t5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.X2I_mllm, torch_dtype=torch.bfloat16)
        tokenizer_t5 = AutoProcessor.from_pretrained(args.X2I_mllm, padding_side="left")
    
        proj_t5 = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True)
        state_dict = torch.load(args.X2I_alignet, map_location="cpu")
        state_dict_new = {}
        for k,v in state_dict.items():
            k_new = k.replace("module.","")
            state_dict_new[k_new] = v
        proj_t5.load_state_dict(state_dict_new)
    
        # Load scheduler and models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        # text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    proj_t5.requires_grad_(True)
    
    # text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, dtype=weight_dtype)
    proj_t5.to(accelerator.device, dtype=weight_dtype)
    
    # text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    #### lora_layers ####
    num_experts=args.experts

    task_embeddings_lora = nn.Embedding(args.tasks, 3072)
    task_embeddings_lora.requires_grad_(True)
    task_embeddings_lora.to(accelerator.device, dtype=weight_dtype)


    if args.pretrained_lora_path is not None:
        lora_path = args.pretrained_lora_path

        proj_t5_save_path = lora_path.replace("lora.safetensors","diffusion_pytorch_model.bin")
        state_dict = torch.load(proj_t5_save_path, map_location="cpu")
        state_dict_new = {}
        for k,v in state_dict.items():
            k_new = k.replace("module.","")
            state_dict_new[k_new] = v
        proj_t5.load_state_dict(state_dict_new)

        task_embeddings_lora_path = proj_t5_save_path.replace("diffusion_pytorch_model.bin","task_embeddings.bin")
        state_dict = torch.load(task_embeddings_lora_path, map_location="cpu")
        state_dict_new = {}
        for k,v in state_dict.items():
            k_new = k.replace("module.","")
            state_dict_new[k_new] = v
        task_embeddings_lora.load_state_dict(state_dict_new)
            
        checkpoint = load_safetensors(lora_path)
        
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        number = 1
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                            lora_state_dicts[key] = value
                
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, rank=args.rank, network_alpha=args.network_alpha, device=accelerator.device, dtype=weight_dtype, num_experts=num_experts
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    for i in range(num_experts):
                        lora_attn_procs[name].q_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].q_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.up.weight', None)

                        lora_attn_procs[name].k_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].k_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.up.weight', None)

                        lora_attn_procs[name].v_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].v_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.up.weight', None)

                        lora_attn_procs[name].proj_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].proj_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].q_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.q_moe_lora.gate.weight', None)
                    lora_attn_procs[name].k_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.k_moe_lora.gate.weight', None)
                    lora_attn_procs[name].v_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.v_moe_lora.gate.weight', None)
                    lora_attn_procs[name].proj_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.proj_moe_lora.gate.weight', None)

                    lora_attn_procs[name].q_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.down.weight', None)
                    lora_attn_procs[name].k_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.down.weight', None)
                    lora_attn_procs[name].v_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.down.weight', None)
                    lora_attn_procs[name].proj_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.shared_experts.down.weight', None)

                    lora_attn_procs[name].q_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.up.weight', None)
                    lora_attn_procs[name].k_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.up.weight', None)
                    lora_attn_procs[name].v_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.up.weight', None)
                    lora_attn_procs[name].proj_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.shared_experts.up.weight', None)


                
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                # print("setting LoRA Processor for", name)        
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, rank=args.rank, network_alpha=args.network_alpha, device=accelerator.device, dtype=weight_dtype, num_experts=num_experts
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    for i in range(num_experts):
                        lora_attn_procs[name].q_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].q_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.up.weight', None)

                        lora_attn_procs[name].k_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].k_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.up.weight', None)

                        lora_attn_procs[name].v_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.down.weight', None)
                        lora_attn_procs[name].v_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.up.weight', None)


                    lora_attn_procs[name].q_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.q_moe_lora.gate.weight', None)
                    lora_attn_procs[name].k_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.k_moe_lora.gate.weight', None)
                    lora_attn_procs[name].v_moe_lora.gate.weight.data[:num_experts] = lora_state_dicts.get(f'{name}.v_moe_lora.gate.weight', None)

                    lora_attn_procs[name].q_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.down.weight', None)
                    lora_attn_procs[name].k_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.down.weight', None)
                    lora_attn_procs[name].v_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.down.weight', None)

                    lora_attn_procs[name].q_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.up.weight', None)
                    lora_attn_procs[name].k_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.up.weight', None)
                    lora_attn_procs[name].v_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.up.weight', None)

            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()
    else:
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, rank=args.rank, network_alpha=args.network_alpha, device=accelerator.device, \
                        dtype=weight_dtype,  num_experts=num_experts)
            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, rank=args.rank, network_alpha=args.network_alpha, device=accelerator.device, \
                        dtype=weight_dtype,  num_experts=num_experts)
            else:
                lora_attn_procs[name] = attn_processor        
    ######################
    transformer.set_attn_processor(lora_attn_procs)
    transformer.train()

    transformer_proj = disp_loss(transformer,proj_t5,task_embeddings_lora,use_Dispersive_Loss=True)

    params=[]
    for n, param in transformer.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        else:
            params.append(param) 
    for name, p in proj_t5.named_parameters():
        params.append(p) 

    for name, p in task_embeddings_lora.named_parameters():
        params.append(p) 
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(path.split("-")[-1])
        initial_global_step = global_step
    else:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    # params_to_optimize = [p for p in transformer.parameters() if p.requires_grad] + [p for p in proj_t5.parameters()]
    transformer_parameters_with_lr = {"params": params, "lr": args.learning_rate}

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [transformer_parameters_with_lr],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # 数据准备
    datamodule = DataModuleCustom(
        args, 
        tokenizer_t5=tokenizer_t5, 
        tokenizer_t5_en=tokenizer_two,
        tokenizer_en=tokenizer_one,
    )
    train_dataloader = datamodule._train_dataloader()

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = 10e10
    if args.resume_from_checkpoint:
        first_epoch = global_step // num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer_proj, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer_proj, optimizer, train_dataloader, lr_scheduler
    )
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "Easy_Control"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # some fixed parameters 
    vae_scale_factor = 16
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                with torch.no_grad():
                    inputs = {
                        "input_ids": batch["input_ids_t5"].to(device=accelerator.device),
                        "attention_mask": batch["attention_mask"].to(device=accelerator.device)
                        # "image_grid_thw": batch["image_grid_thw"].to(device=accelerator.device),
                        # "pixel_values": batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype).squeeze(1)
                    }
    
                    output_hidden_state_all = text_encoder_t5.generate(**inputs, max_new_tokens=1,output_hidden_states=True,return_dict_in_generate=True)
                    text_embeddings = torch.stack(output_hidden_state_all["hidden_states"][0], dim=1)

                    # tokens = [batch["text_ids_1"], batch["text_ids_2"]]
                    # prompt_embeds, pooled_prompt_embeds, text_ids = encode_token_ids(text_encoders, tokens, accelerator)
                    # prompt_embeds = prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                    # pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
                    # text_ids = text_ids.to(dtype=vae.dtype, device=accelerator.device)
                    
                    pixel_values = batch["pixel_values_output"].to(dtype=vae.dtype, device=accelerator.device)
                    height_ = 2 * (int(pixel_values.shape[-2]) // vae_scale_factor)
                    width_ = 2 * (int(pixel_values.shape[-1]) // vae_scale_factor)
    
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)
    
                    latent_image_ids, cond_latent_image_ids = resize_position_encoding(
                        model_input.shape[0],
                        height_,
                        width_,
                        height_,
                        width_,
                        accelerator.device,
                        weight_dtype,
                    )
    
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
    
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    
                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    
                    packed_noisy_model_input = FluxPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )
                    
                    latent_image_ids_to_concat = [latent_image_ids]
                    packed_cond_model_input_to_concat = []
                    
                    subject_pixel_values = batch["pixel_values_input"].to(device=accelerator.device, dtype=vae.dtype)
                    subject_input = vae.encode(subject_pixel_values).latent_dist.sample()
                    subject_input = (subject_input - vae_config_shift_factor) * vae_config_scaling_factor
                    subject_input = subject_input.to(dtype=weight_dtype)             
                    latent_subject_ids = prepare_latent_subject_ids(height_, width_, accelerator.device, weight_dtype)
                    latent_subject_ids[:, 1] += args.offset
                    sub_latent_image_ids = torch.concat([latent_subject_ids], dim=-2)
                    latent_image_ids_to_concat.append(sub_latent_image_ids)
                    
                    packed_subject_model_input = FluxPipeline._pack_latents(    
                        subject_input,
                        batch_size=subject_input.shape[0],
                        num_channels_latents=subject_input.shape[1],
                        height=subject_input.shape[2],
                        width=subject_input.shape[3],
                    )
                    packed_cond_model_input_to_concat.append(packed_subject_model_input)
                

                    
                    latent_image_ids = torch.concat(latent_image_ids_to_concat, dim=-2)
                    cond_packed_noisy_model_input = torch.concat(packed_cond_model_input_to_concat, dim=-2)
    
                    # handle guidance
                    if accelerator.unwrap_model(transformer).config.guidance_embeds:
                        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                        guidance = guidance.expand(model_input.shape[0])
                    else:
                        guidance = None

                task_type = batch["task"].to(dtype=weight_dtype, device=accelerator.device)

                model_pred, hidden_states_all= transformer_proj(
                    task_type=task_type,
                    packed_noisy_model_input=packed_noisy_model_input,
                    cond_packed_noisy_model_input=cond_packed_noisy_model_input,
                    timesteps=timesteps,
                    guidance=guidance,
                    text_embeddings=text_embeddings,
                    latent_image_ids=latent_image_ids
                )

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(pixel_values.shape[-2]),
                    width=int(pixel_values.shape[-1]),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )

                loss = loss.mean()

                del packed_noisy_model_input
                del cond_packed_noisy_model_input
                del timesteps
                del guidance
                del text_embeddings
                del latent_image_ids
                del model_pred
                gc.collect()

                # Compute Dispersive Loss. gathered_list = [torch.zeros_like(hidden_states_all) for _ in range(8)]  # 预分配缓冲区
                if hidden_states_all:
                    # print(task_type)
                    loss_dispersive = transformer_proj.module.compute_disp_loss_all(hidden_states_all, task_type)
                    loss += loss_dispersive

                del task_type
                del hidden_states_all
                gc.collect()

                accelerator.backward(loss)
                gc.collect()

                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        state_dict = proj_t5.state_dict()
                        torch.save(state_dict, os.path.join(save_path, "diffusion_pytorch_model.bin"))

                        state_dict = task_embeddings_lora.state_dict()
                        torch.save(state_dict, os.path.join(save_path, "task_embeddings.bin"))

                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
                        lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)


    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    main(args)

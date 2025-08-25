import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import copy
import deepspeed
import re
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from src.custom_dataset_qwen import DataModuleCustom 

from einops import rearrange
from accelerate.state import AcceleratorState

from typing import Callable, List, Optional, Union
from transformers import T5Tokenizer,MT5EncoderModel,AutoModel,AutoModelForCausalLM
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers import T5ForConditionalGeneration,AutoTokenizer
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.loaders.lora_pipeline import SD3LoraLoaderMixin
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers.utils import ContextManagers
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from diffusers.utils import get_peft_kwargs,get_adapter_name

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
)
from src.layers_qwen import MultiDoubleStreamBlockLoraProcessor
from src.transformer_qwen import QwenImageTransformer2DModel
from torch import nn
from safetensors.torch import save_file
from torchvision.utils import save_image

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/workspace/group/models/Qwen-Image",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # parser.add_argument(
    #     "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    # )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
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
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

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
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser = DataModuleCustom.add_data_specific_args(parser)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank



    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


def get_sigmas(timesteps,noise_scheduler_copy, n_dim=4, device="cuda", dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]



class disp_loss(torch.nn.Module):
    def __init__(self, transformer, task_embeddings_lora, use_Dispersive_Loss=False):
        super().__init__()
        self.transformer = transformer
        self.task_embeddings_lora = task_embeddings_lora
        self.use_Dispersive_Loss = use_Dispersive_Loss

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


    def forward(self,task_type,packed_noisy_model_input,prompt_embeds,encoder_attention_mask,
                timesteps,img_shapes_new):

        task_embeddings = self.task_embeddings_lora(task_type)      
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=encoder_attention_mask,
            timestep=timesteps / 1000,
            img_shapes=img_shapes_new,
            txt_seq_lens=encoder_attention_mask.sum(dim=1).tolist(),
            return_dict=False,
            task_embeddings=task_embeddings,
        )[0]

        if self.use_Dispersive_Loss:
            return model_pred, hidden_states_all
        else:
            return model_pred, None

def main(args):

    weight_dtype = torch.bfloat16
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    teacher = args.pretrained_model_name_or_path
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(teacher, subfolder="scheduler", shift=3.0)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(
        teacher,
        subfolder="vae",
        # revision=args.revision,
        # variant=args.variant,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        teacher, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    tokenizer_qwen = Qwen2Tokenizer.from_pretrained(teacher ,subfolder="tokenizer")

    transformer = QwenImageTransformer2DModel.from_pretrained(
        teacher,
        subfolder="transformer",
        # revision=args.revision,
        # variant=args.variant,
        # quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )


    lora_attn_procs = {}
    double_blocks_idx = list(range(60))
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            # print("setting LoRA Processor for", name)
            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, rank=args.rank, network_alpha=args.network_alpha, device=accelerator.device, \
                    dtype=weight_dtype,  num_experts=args.experts)
        else:
            lora_attn_procs[name] = attn_processor        
    ######################
    transformer.set_attn_processor(lora_attn_procs)

    # Freeze vae and text encoders.
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    text_encoder.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = {"device": accelerator.device, "dtype": weight_dtype}
    transformer.to(**transformer_to_kwargs)

    task_embeddings_lora = nn.Embedding(args.tasks, 3072)
    task_embeddings_lora.requires_grad_(True)
    task_embeddings_lora.to(accelerator.device, dtype=weight_dtype)

    transformer_proj = disp_loss(transformer,task_embeddings_lora,use_Dispersive_Loss=True)


    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        # optimizer_class = adamw.AdamW
        # optimizer_class = adamw2.AdamW

    params=[]
    for n, param in transformer.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        else:
            params.append(param) 

    for name, p in task_embeddings_lora.named_parameters():
        params.append(p) 
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    optimizer = optimizer_class(
                    iter(params),
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                    # threshold=2e-3,
                    # progressive_iter=2,
                    # lambda_rank=0.0005,
                    )


    datamodule = DataModuleCustom(args, tokenizer_qwen=tokenizer_qwen)
    train_dataloader = datamodule._train_dataloader()
    # len_train_dataloader = len(datamodule.datasets['train'])//datamodule.batch_size # len(train_dataloader)

    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = 10e10
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=args.batch_size
    transformer_proj, optimizer,train_dataloader = accelerator.prepare(transformer_proj,optimizer,train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    ############## Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    transformer.enable_gradient_checkpointing()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer_proj):
                with torch.no_grad():
                    pixel_values = batch["pixel_values_output"].to(dtype=vae.dtype, device=accelerator.device).unsqueeze(2)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - latents_mean) * latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                    pixel_values_in = batch["pixel_values_input"].to(dtype=vae.dtype, device=accelerator.device).unsqueeze(2)
                    model_input_in = vae.encode(pixel_values_in).latent_dist.sample()
                    model_input_in = (model_input_in - latents_mean) * latents_std
                    model_input_in = model_input_in.to(dtype=weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=0,
                        logit_std=1,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                     
                    sigmas = get_sigmas(timesteps,noise_scheduler_copy, n_dim=model_input.ndim, device=accelerator.device, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    # Predict the noise residual
                    img_shapes = [
                        (1, args.resolution // vae_scale_factor // 2, args.resolution // vae_scale_factor // 2)
                    ] * bsz
                    # transpose the dimensions
                    noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
                    packed_noisy_model_input = QwenImagePipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[3],
                        width=model_input.shape[4],
                    )
                    ## model_input_in.shape torch.Size([2, 16, 1, 64, 64])
                    packed_cond_input = QwenImagePipeline._pack_latents(
                        model_input_in.permute(0, 2, 1, 3, 4),
                        batch_size=model_input_in.shape[0],
                        num_channels_latents=model_input_in.shape[1],
                        height=model_input_in.shape[3],
                        width=model_input_in.shape[4],
                    )
                    
                    encoder_hidden_states = text_encoder(
                        input_ids=batch["input_ids"].to(device=accelerator.device),
                        attention_mask=batch["attention_mask"].to(device=accelerator.device),
                        output_hidden_states=True,
                    )
                    hidden_states = encoder_hidden_states.hidden_states[-1]
                    split_hidden_states = _extract_masked_hidden(hidden_states, batch["attention_mask"].to(device=accelerator.device))
                    split_hidden_states = [e[64:] for e in split_hidden_states]
                    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=accelerator.device) for e in split_hidden_states]
                    max_seq_len = max([e.size(0) for e in split_hidden_states])
                    prompt_embeds = torch.stack(
                        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
                    )
                    encoder_attention_mask = torch.stack(
                        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
                    )

                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
                    # print(f"{encoder_attention_mask.sum(dim=1).tolist()=}")
                    
                    #### subject_input
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_input], dim=1)

                task_type = batch["task"].to(dtype=torch.int, device=accelerator.device)                
                # img_shapes_new = [(1, args.resolution // vae_scale_factor//2, args.resolution // vae_scale_factor )] * bsz # [(1, 32, 64)]
                # img_shapes_new = [(2, args.resolution // vae_scale_factor//2, args.resolution // vae_scale_factor//2 )] * bsz # [(2, 32, 32)]
                img_shapes_new = [[(1, args.resolution // vae_scale_factor//2, args.resolution // vae_scale_factor//2 ),(1, args.resolution // vae_scale_factor//2, args.resolution // vae_scale_factor//2 )]] * bsz # [(2, 32, 32)]

                model_pred,_ = transformer_proj(task_type,packed_noisy_model_input,prompt_embeds,encoder_attention_mask,timesteps,img_shapes_new)

                model_pred = model_pred[:, :packed_cond_input.shape[1]]
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred, args.resolution, args.resolution, vae_scale_factor
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes 检查梯度当前是否在所有进程之间同步
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        print(f"#########{args.checkpointing_steps} saving model #######")
                        save_path = os.path.join(args.output_dir, f"{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        state_dict = task_embeddings_lora.state_dict()
                        torch.save(state_dict, os.path.join(save_path, "task_embeddings.bin"))

                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
                        lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
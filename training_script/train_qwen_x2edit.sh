# export NCCL_P2P_LEVEL=NVL

# pip install -U deepspeed
# pip install /home/notebook/data/group/group/majian/torch_stable/torch-2.1.2+cu118-cp311-cp311-linux_x86_64.whl
# pip install /home/notebook/data/group/group/majian/torch_stable/torchvision-0.16.2+cu118-cp311-cp311-linux_x86_64.whl
# pip install huggingface_hub==0.24.5
# pip install accelerate==1.2.1
# pip install peft==0.14.0
# pip install bitsandbytes
# pip install diffusers==0.30.3
# pip install piq
# pip install tokenizers==0.21.0
pip install peft==0.17.0
pip install transformers==4.54.1


DATA_ARGS="--webdataset_base_urls \
        /home/notebook/data/group/group/text2img_data/X2I/tars/Kontext/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/Kontext_aspect/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/Kontext_aspect1/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/ShareGPT-4o-Image/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/ShareGPT-4o-Image/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/Kontext_subject/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/subject/Subjects200K/1/* \
        /home/notebook/data/group/group/XJZhu/textflux-main/textflux-qwen-generate-en/*/* \
        /home/notebook/data/group/group/XJZhu/textflux-main/textflux-qwen-generate-ch/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/bagel_stage1/*/* \
        /home/notebook/data/group/group/text2img_data/X2I/tars/bagel_reason/*/* \
        --pretrained_model_name_or_path /home/notebook/data/group/group/models/Qwen-Image \
        --num_workers 2 \
        --batch_size 12 \
        --shard_width 5 \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        "

MODEL_ARGS="\
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000000 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --mixed_precision="bf16" \
  --checkpointing_steps=1000 \
  --output_dir="/home/notebook/data/group/group/majian/X2Edit/models/lora_32" \
  --max_grad_norm=1 \
  --checkpoints_total_limit=5 \
  "

export options="\
      $DATA_ARGS\
      $MODEL_ARGS"

export CC=gcc
export CXX=g++

accelerate launch --config_file "ds_config.yaml" --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_machines  $WORLD_SIZE  --num_processes 32 train_qwen_x2edit.py $options
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file "ds_config.yaml" --machine_rank 0 --main_process_ip  127.0.0.1 --main_process_port 29000 --num_machines  1  --num_processes 8 train_qwen_x2edit.py $options

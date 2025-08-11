export CONFIG="./ds_config.yaml"
RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=27001
WORLD_SIZE=1
NUM_PROCESSES=2
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file $CONFIG --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_machines  $WORLD_SIZE --num_processes $NUM_PROCESSES train_512.py \
    --pretrained_model_name_or_path /mnt/workspace/group/models/flux/FLUX.1-dev \
    --webdataset_base_urls /mnt/workspace/group/text2img_data/X2I/tars/Kontext/*/* \
    /mnt/workspace/group/text2img_data/X2I/tars/Kontext_aspect/*/* \
    /mnt/workspace/group/text2img_data/X2I/tars/bagel_reason/*/* \
    /mnt/workspace/group/text2img_data/X2I/tars/bagel_stage1/*/* \
    /mnt/workspace/group/text2img_data/X2I/tars/Kontext_aspect1/*/* \
    --rank 64 \
    --network_alpha 64 \
    --tasks 16 \
    --experts 12 \
    --offset 64 \
    --output_dir ./models \
    --mixed_precision="bf16" \
    --learning_rate=1e-4 \
    --batch_size 12 \
    --num_train_epochs=1000 \
    --checkpointing_steps=400 \
    --gradient_checkpointing \
    --X2I_mllm /mnt/workspace/group/models/Qwen2.5-VL-7B-Instruct \
    --X2I_alignet /mnt/workspace/group/**/flux/result_fit_speed/qwenvl25_dev_norm/57000/diffusion_pytorch_model.bin


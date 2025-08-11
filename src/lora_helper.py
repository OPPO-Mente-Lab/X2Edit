from diffusers.models.attention_processor import FluxAttnProcessor2_0
from safetensors import safe_open
import re
import torch
from src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    return checkpoint

def update_model_with_lora(checkpoint, transformer, num_experts,device):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
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
                
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=rank, device=device, dtype=torch.bfloat16, num_experts=num_experts
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for i in range(num_experts):
                    lora_attn_procs[name].q_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].q_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].k_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].k_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].v_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].v_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].proj_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].proj_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.experts.{i}.up.weight', None)


                    # lora_attn_procs[name].q_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].q_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gates.{i}.up.weight', None)

                    # lora_attn_procs[name].k_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].k_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gates.{i}.up.weight', None)

                    # lora_attn_procs[name].v_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].v_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gates.{i}.up.weight', None)

                    # lora_attn_procs[name].proj_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].proj_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.feature_gates.{i}.up.weight', None)

                lora_attn_procs[name].q_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.gate.weight', None)
                lora_attn_procs[name].k_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.gate.weight', None)
                lora_attn_procs[name].v_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.gate.weight', None)
                lora_attn_procs[name].proj_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.gate.weight', None)

                lora_attn_procs[name].q_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.down.weight', None)
                lora_attn_procs[name].k_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.down.weight', None)
                lora_attn_procs[name].v_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.down.weight', None)
                lora_attn_procs[name].proj_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.shared_experts.down.weight', None)

                lora_attn_procs[name].q_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.up.weight', None)
                lora_attn_procs[name].k_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.up.weight', None)
                lora_attn_procs[name].v_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.up.weight', None)
                lora_attn_procs[name].proj_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.shared_experts.up.weight', None)


                # lora_attn_procs[name].q_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gate_shared.down.weight', None)
                # lora_attn_procs[name].k_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gate_shared.down.weight', None)
                # lora_attn_procs[name].v_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gate_shared.down.weight', None)
                # lora_attn_procs[name].proj_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.feature_gate_shared.down.weight', None)

                # lora_attn_procs[name].q_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gate_shared.up.weight', None)
                # lora_attn_procs[name].k_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gate_shared.up.weight', None)
                # lora_attn_procs[name].v_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gate_shared.up.weight', None)
                # lora_attn_procs[name].proj_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.proj_moe_lora.feature_gate_shared.up.weight', None)


                lora_attn_procs[name].to(device,dtype=transformer.dtype) 


            elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[key] = value
                
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=rank, device=device, dtype=torch.bfloat16, num_experts=num_experts
                )
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for i in range(num_experts):
                    lora_attn_procs[name].q_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].q_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].k_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].k_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.experts.{i}.up.weight', None)

                    lora_attn_procs[name].v_moe_lora.experts[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.down.weight', None)
                    lora_attn_procs[name].v_moe_lora.experts[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.experts.{i}.up.weight', None)


                    # lora_attn_procs[name].q_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].q_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gates.{i}.up.weight', None)

                    # lora_attn_procs[name].k_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].k_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gates.{i}.up.weight', None)

                    # lora_attn_procs[name].v_moe_lora.feature_gates[i].down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gates.{i}.down.weight', None)
                    # lora_attn_procs[name].v_moe_lora.feature_gates[i].up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gates.{i}.up.weight', None)

                    
                lora_attn_procs[name].q_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.gate.weight', None)
                lora_attn_procs[name].k_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.gate.weight', None)
                lora_attn_procs[name].v_moe_lora.gate.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.gate.weight', None)

                lora_attn_procs[name].q_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.down.weight', None)
                lora_attn_procs[name].k_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.down.weight', None)
                lora_attn_procs[name].v_moe_lora.shared_experts.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.down.weight', None)

                lora_attn_procs[name].q_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.shared_experts.up.weight', None)
                lora_attn_procs[name].k_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.shared_experts.up.weight', None)
                lora_attn_procs[name].v_moe_lora.shared_experts.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.shared_experts.up.weight', None)


                # lora_attn_procs[name].q_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gate_shared.down.weight', None)
                # lora_attn_procs[name].k_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gate_shared.down.weight', None)
                # lora_attn_procs[name].v_moe_lora.feature_gate_shared.down.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gate_shared.down.weight', None)

                # lora_attn_procs[name].q_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.q_moe_lora.feature_gate_shared.up.weight', None)
                # lora_attn_procs[name].k_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.k_moe_lora.feature_gate_shared.up.weight', None)
                # lora_attn_procs[name].v_moe_lora.feature_gate_shared.up.weight.data = lora_state_dicts.get(f'{name}.v_moe_lora.feature_gate_shared.up.weight', None)


                lora_attn_procs[name].to(device,dtype=transformer.dtype) 
            else:
                lora_attn_procs[name] = FluxAttnProcessor2_0()

        transformer.set_attn_processor(lora_attn_procs)
        

def set_single_lora(transformer, local_path, num_experts=6,device="cuda"):
    checkpoint = load_checkpoint(local_path)
    update_model_with_lora(checkpoint,  transformer,num_experts,device)


def unset_lora(transformer):
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        lora_attn_procs[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(lora_attn_procs)



import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention

from typing import Optional
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch.distributed.nn.functional import all_gather


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h) #2048,6144
        logits = F.linear(hidden_states, self.weight, None) #12,6144
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                # aux_loss = (Pi * fi).sum() * self.alpha
                aux_loss = None
                # save_load_balancing_loss((aux_loss, Pi, fi, self.alpha))
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoELoRALayer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        rank: int,
        network_alpha: int,
        num_activated_experts=2,
        task_guide=True
    ):
        super().__init__()
        self.shared_experts = LoRALinearLayer(dim, hidden_dim,rank,network_alpha)
        self.experts = nn.ModuleList([LoRALinearLayer(dim, hidden_dim,rank,network_alpha) for i in range(num_routed_experts)])
        if task_guide:
            self.gate = MoEGate(
                embed_dim = dim+hidden_dim, 
                num_routed_experts = num_routed_experts, 
                num_activated_experts = num_activated_experts
            )
        else:
            self.gate = MoEGate(
                embed_dim = dim, 
                num_routed_experts = num_routed_experts, 
                num_activated_experts = num_activated_experts
            )
        self.num_activated_experts = num_activated_experts

    def forward(self, x, task_embed_ori=None):
        batch_size, seq_len, _ = x.size()
        # 获取任务嵌入并扩展至序列长度维度
        if task_embed_ori is not None:
            if len(task_embed_ori.shape)==1:
                task_embed = task_embed_ori.unsqueeze(0).unsqueeze(0).expand(-1, seq_len, -1) 
            else:
                task_embed = task_embed_ori.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, task_embed_dim)
                
            combined_input = torch.cat([x, task_embed], dim=-1)  # (batch, seq_len, in_dim + task_embed_dim)
        else:
            combined_input = x
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(combined_input) 
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape).to(dtype=wtype)
            #y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        number=0,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.number = number

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank,network_alpha, device=None, dtype=None, num_experts=4):
        super().__init__()
        # Initialize a list to store the LoRA layers

        self.q_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)
        self.k_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)
        self.v_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)


    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        task_embeddings: Optional[torch.Tensor] = None,
        use_cond = False,
    ) -> torch.FloatTensor:
                
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        query = attn.to_q(hidden_states) # 1*2560*3072
        key = attn.to_k(hidden_states) 
        value = attn.to_v(hidden_states) 
        
        query = query + self.q_moe_lora(hidden_states,task_embeddings)
        key = key + self.k_moe_lora(hidden_states,task_embeddings)
        value = value + self.v_moe_lora(hidden_states,task_embeddings)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        block_size = int((hidden_states.shape[1]-512)/2) + 512

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, : block_size,:]

        return hidden_states if not use_cond else (hidden_states, cond_hidden_states)


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank,network_alpha, device=None, dtype=None,num_experts=4):
        super().__init__()

        self.q_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)
        self.k_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)
        self.v_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)
        self.proj_moe_lora = MoELoRALayer(dim, dim, num_experts, rank,network_alpha)

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        task_embeddings: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `context` projections.
        inner_dim = 3072
        head_dim = inner_dim // attn.heads
        
        query = attn.to_q(hidden_states) # 1*2048*3072
        key = attn.to_k(hidden_states) 
        value = attn.to_v(hidden_states) 

        query = query + self.q_moe_lora(hidden_states,task_embeddings)
        key = key + self.k_moe_lora(hidden_states,task_embeddings)
        value = value + self.v_moe_lora(hidden_states,task_embeddings)

        # encoder_hidden_states[:,-1] =  task_embeddings
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) # 1*512*3072 -->  1*512*3072
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)


        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)



        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        block_size = int((hidden_states.shape[1])/2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = hidden_states + self.proj_moe_lora(hidden_states,task_embeddings)
        # encoder_hidden_states = encoder_hidden_states + self.proj_moe_lora_t(encoder_hidden_states,task_embeddings)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, :block_size,:]
        
        return (hidden_states, encoder_hidden_states, cond_hidden_states,task_embeddings) if use_cond else (encoder_hidden_states, hidden_states)
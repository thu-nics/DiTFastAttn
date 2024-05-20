import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn
import torch.nn as nn

class FastAttnProcessor:
    def __init__(self, window_size,steps_method):
        self.window_size=window_size
        self.steps_method=steps_method
        self.need_compute_residual=self.compute_need_compute_residual(steps_method)
        self.need_cache_output=True
        # print(f"need_compute_residual {[(_,__) for _,__ in zip(steps_method,self.need_compute_residual)]}")
        
    
    def compute_need_compute_residual(self,steps_method):
        need_compute_residual=[]
        for i,method in enumerate(steps_method):
            need=False
            if "full_attn" in method:
                for j in range(i+1,len(steps_method)):
                    if "residual_window_attn" in steps_method[j]:
                        need=True
                    if "full_attn" in steps_method[j]:
                        break
            need_compute_residual.append(need)
        return need_compute_residual
    
    def run_forward_method(self,attn,hidden_states,encoder_hidden_states,attention_mask,temb,method):
        residual = hidden_states
        if method=="output_share":
            hidden_states = attn.cached_output
        else:
            
            if "cfg_attn_share" in method:
                batch_size=hidden_states.shape[0]
                hidden_states = hidden_states[:batch_size//2]
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states[:batch_size//2]
                if attention_mask is not None:
                    attention_mask = attention_mask[:batch_size//2]
            
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            
            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)

            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)
            
            if attention_mask is not None:
                assert "residual_window_attn" not in method
                
                hidden_states = F.scaled_dot_product_attention(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                ).transpose(1, 2)
            elif "full_attn" in method:
                all_hidden_states=flash_attn.flash_attn_func(query, key, value)
                if self.need_compute_residual[attn.stepi]:
                    # w_hidden_states = self.window_attn(
                    #     query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                    # ).transpose(1, 2)
                    w_hidden_states=flash_attn.flash_attn_func(query, key, value,window_size=self.window_size)
                    residual=all_hidden_states-w_hidden_states
                    if "cfg_attn_share" in method:
                        residual=torch.cat([residual, residual], dim=0)
                    attn.cached_residual=residual
                hidden_states=all_hidden_states
            elif "residual_window_attn" in method:
                # w_hidden_states = self.window_attn(
                #         query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                #     ).transpose(1, 2)
                w_hidden_states=flash_attn.flash_attn_func(query, key, value,window_size=self.window_size)
                hidden_states=w_hidden_states+attn.cached_residual[:batch_size].view_as(w_hidden_states)
            

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            

            if "cfg_attn_share" in method:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            
            if self.need_cache_output:
                attn.cached_output = hidden_states
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states=self.run_forward_method(attn,hidden_states,encoder_hidden_states,attention_mask,temb,self.steps_method[attn.stepi])
        attn.stepi += 1
        return hidden_states

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn
import functools
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np
import os

class FastAttnProcessor:
    def __init__(self, window_size,steps_method):
        self.window_size=window_size
        self.steps_method=steps_method
    
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
        residual = hidden_states
        method=self.steps_method[attn.stepi]
        
        if method=="output_share":
            hidden_states = attn.cached_output
        else:
        
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            assert attention_mask is None
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            
            if "cfg_attn_share" in method:
                query = attn.to_q(hidden_states[:batch_size//2])
                #TODO replace flash-attn implmentation
                query=torch.cat([query,query],dim=0)
                
            else:
                query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            if "cfg_attn_share" in method:
                key = attn.to_k(encoder_hidden_states[:batch_size//2])
                #TODO replace flash-attn implmentation
                key=torch.cat([key,key],dim=0)
            else:
                key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)

            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)
            
            if "full_attn" in method:
                all_hidden_states=flash_attn.flash_attn_func(query, key, value)
                w_hidden_states=flash_attn.flash_attn_func(query, key, value,window_size=self.window_size)
                residual=all_hidden_states-w_hidden_states
                attn.cached_residual=residual
                hidden_states=all_hidden_states
            elif "residual_window_attn" in method:
                w_hidden_states=flash_attn.flash_attn_func(query, key, value,window_size=self.window_size)
                hidden_states=w_hidden_states+attn.cached_residual.view_as(w_hidden_states)
            

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            attn.cached_output = hidden_states

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        attn.stepi += 1
        return hidden_states

def set_stepi_warpper(pipe):
    @functools.wraps(pipe)
    def wrapper(*args, **kwargs):
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            block.attn1.stepi=0
            block.attn1.cached_residual=None
            block.attn1.cached_output=None
            # print(f"Reset stepi to 0 for block {blocki}")
        return pipe(*args, **kwargs)

    return wrapper


def transform_model_fast_attention(raw_pipe, n_steps, n_calib, threshold=0.98, window_size=[-64,64],use_cache=False,seed=3,independent_calib=False):
    pipe = set_stepi_warpper(raw_pipe)
    
    cache_file=f"cache/{raw_pipe.config._name_or_path.replace('/','_')}_{n_steps}_{n_calib}.pt"
    if use_cache and os.path.exists(cache_file):
        all_steps_method=torch.load(cache_file)
    else:
        # calibration
        
        # reset all processors
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            attn: Attention = block.attn1
            attn.set_processor(AttnProcessor2_0())
        
        # calibration raw
        np.random.seed(seed)
        x=torch.randint(0, 1000, (n_calib,)).cuda()
        generator=torch.manual_seed(seed)
        raw_outs=pipe(x,num_inference_steps=n_steps,generator=generator,output_type='np.array',return_dict=False)
        raw_outs=np.concatenate(raw_outs,axis=0)
        
        all_steps_method=[]
        
        # sequential greedy calibration 
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            if 1 and blocki==1:
                # DEBUG
                all_steps_method*=len(pipe.transformer.transformer_blocks)
                break
            steps_method=["full_attn"]*n_steps
            for step_i in range(1, n_steps):
                selected_method="full_attn"
                for method in ["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]:
                    steps_method[step_i]=method
                    # TODO search widnow_size
                    processor=FastAttnProcessor(window_size,steps_method)
                    attn.set_processor(processor)
                    # compute output
                    generator=torch.manual_seed(seed)
                    outs=pipe(x,num_inference_steps=n_steps,generator=generator,output_type='np.array',return_dict=False)
                    outs=np.concatenate(outs,axis=0)
                    ssim=0
                    for i in range(raw_outs.shape[0]):
                        ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
                    ssim/=raw_outs.shape[0]
                    print(f"Block {blocki} step {step_i} method={method} SSIM {ssim}" )
                    if ssim>threshold:
                        selected_method=method
                    del processor
                steps_method[step_i]=selected_method
            print(f"Block {blocki} selected steps_method {steps_method}")
            if independent_calib:
                # independent calibration
                processor=AttnProcessor2_0()
            else:
                processor=FastAttnProcessor(window_size,steps_method)
                
            attn.set_processor(processor)
            all_steps_method.append(steps_method)
        
        # save cache
        if not os.path.exists("cache"):
            os.makedirs("cache")
        torch.save(all_steps_method,cache_file)
    
    # set processor
    for blocki, block in enumerate(pipe.transformer.transformer_blocks):
        attn: Attention = block.attn1
        attn.set_processor(FastAttnProcessor(window_size,all_steps_method[blocki]))
        
    # test final ssim
    generator=torch.manual_seed(seed)
    outs=pipe(x,num_inference_steps=n_steps,generator=generator,output_type='np.array',return_dict=False)
    outs=np.concatenate(outs,axis=0)
    ssim=0
    for i in range(raw_outs.shape[0]):
        ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
    ssim/=raw_outs.shape[0]
    print(f"Final SSIM {ssim}")

    return pipe,ssim
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn
import functools
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import collections
import numpy as np
import os
import time

class FastAttnProcessor:
    def __init__(self, window_size,steps_method):
        self.window_size=window_size
        self.steps_method=steps_method
        self.need_compute_residual=self.compute_need_compute_residual(steps_method)
        # print(f"need_compute_residual {[(_,__) for _,__ in zip(steps_method,self.need_compute_residual)]}")
        
        self.is_calibration=False
        self.calib_cos_sim_threshold=None
    
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
        if method=="output_share":
            hidden_states = attn.cached_output
        else:
            
            if "cfg_attn_share" in method:
                batch_size=hidden_states.shape[0]
                hidden_states = hidden_states[:batch_size//2]
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states[:batch_size//2]
            
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
            
            if "full_attn" in method:
                all_hidden_states=flash_attn.flash_attn_func(query, key, value)
                if self.need_compute_residual[attn.stepi]:
                    w_hidden_states=flash_attn.flash_attn_func(query, key, value,window_size=self.window_size)
                    residual=all_hidden_states-w_hidden_states
                    if "cfg_attn_share" in method:
                        residual=torch.cat([residual, residual], dim=0)
                    attn.cached_residual=residual
                hidden_states=all_hidden_states
            elif "residual_window_attn" in method:
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
            
            attn.cached_output = hidden_states
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
        residual = hidden_states
        method=self.steps_method[attn.stepi]
        if self.is_calibration:
            if attn.stepi!=0:
                raw_out=self.run_forward_method(attn,hidden_states,encoder_hidden_states,attention_mask,temb,"full_attn")
                batch_size=hidden_states.shape[0]
                selected_method="full_attn"
                for test_mehtod in ["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]:
                    out=self.run_forward_method(attn,hidden_states,encoder_hidden_states,attention_mask,temb,test_mehtod)
                    cos_sim=F.cosine_similarity(raw_out.view(batch_size,-1),out.view(batch_size,-1),dim=1).mean()
                    # print(f"Method {test_mehtod} cos_sim {cos_sim} calib_cos_sim_threshold {self.calib_cos_sim_threshold}")
                    if cos_sim>self.calib_cos_sim_threshold:
                        selected_method=test_mehtod
                # print(f"Selected method {selected_method}")
                self.steps_method[attn.stepi]=selected_method
            self.need_compute_residual[attn.stepi]=True
        hidden_states=self.run_forward_method(attn,hidden_states,encoder_hidden_states,attention_mask,temb,self.steps_method[attn.stepi])
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        attn.stepi += 1
        return hidden_states
    

# def processor_calibration_wrap(processor,cos_sim_threshold):
#     processor.old_call=processor.__call__
#     @functools.wraps(processor.__call__)
#     def wrapper(*args, **kwargs):
#         processor.is_calibration=True
#         stepi=args[0].stepi
#         raw_out=processor(*args, **kwargs)
#         batch_size=raw_out.shape[0]
#         selected_method="full_attn"
#         for mehtod in ["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]:
#             processor.steps_method[stepi]=mehtod
#             processor.compute_need_compute_residual(processor.steps_method)
#             out=processor(*args, **kwargs)
#             cos_sim=F.cosine_similarity(raw_out.view(batch_size,-1),out.view(batch_size,-1),dim=1).mean()
#             if cos_sim>cos_sim_threshold:
#                 selected_method=mehtod
#         print(f"Selected method {selected_method} for step {stepi}")
#         processor.steps_method[stepi]=selected_method
#         processor.compute_need_compute_residual(processor.steps_method)
#         processor.is_calibration=False
#         final_out=processor(*args, **kwargs)
#         processor.is_calibration=True
#         return final_out
#     return wrapper

def set_stepi_warp(pipe):
    @functools.wraps(pipe)
    def wrapper(*args, **kwargs):
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            block.attn1.stepi=0
            block.attn1.cached_residual=None
            block.attn1.cached_output=None
            # print(f"Reset stepi to 0 for block {blocki}")
        return pipe(*args, **kwargs)

    return wrapper
            
def transform_model_fast_attention(raw_pipe, n_steps, n_calib, calib_x, threshold, window_size=[-64,64],use_cache=False,seed=3,sequential_calib=False,debug=False):
    pipe = set_stepi_warp(raw_pipe)
    blocks=pipe.transformer.transformer_blocks
    
    # calibration raw
    generator=torch.manual_seed(seed)
    raw_outs=pipe(calib_x,num_inference_steps=n_steps,generator=generator,output_type='np',return_dict=False)
    raw_outs=np.concatenate(raw_outs,axis=0)
    
    cache_file=f"cache/{raw_pipe.config._name_or_path.replace('/','_')}_{n_steps}_{n_calib}_{threshold}_{sequential_calib}.pt"
    if use_cache and os.path.exists(cache_file):
        all_steps_method=torch.load(cache_file)
    else:
        
        # reset all processors
        for blocki, block in enumerate(blocks):
            attn: Attention = block.attn1
            attn.raw_processor=attn.processor
            attn.set_processor(AttnProcessor2_0())
        
        all_steps_method=[["full_attn" for __ in range(n_steps)] for _ in range(len(blocks))]
        
        # ssim_theshold for each calibration
        ssim_thresholds=[]
        interval=(1-threshold)/len(blocks)/2
        now_threshold=1-interval*len(blocks)
        for blocki in range(len(blocks)):
            now_threshold-=interval
            ssim_thresholds.append(now_threshold)
        print(f"SSIM thresholds {ssim_thresholds}")

        # greedy calibration 
        for blocki, block in enumerate(blocks):
            attn=block.attn1
            # hook to get the input of block
            def hook_fn(module, input, output):
                attn.cached_input=input[0]
            
            # binary search
            cos_sim_threshold_st=0.8
            cos_sim_threshold_ed=1
            n_search=6
            for searchi in range(n_search):
                if searchi==n_search-1:
                    cos_sim_threshold=cos_sim_threshold_ed
                else:
                    cos_sim_threshold=(cos_sim_threshold_st+cos_sim_threshold_ed)/2
                steps_method=["full_attn" for _ in range(n_steps)]
                attn.set_processor(FastAttnProcessor(window_size,steps_method))
                attn.processor.calib_cos_sim_threshold=cos_sim_threshold
                attn.processor.is_calibration=True
                # calib
                outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
                attn.processor.is_calibration=False
                attn.processor.need_compute_residual=attn.processor.compute_need_compute_residual(attn.processor.steps_method)
                # print(f"Block {blocki} need_compute_residual {attn.processor.need_compute_residual}")
                
                # compute output
                # outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
                outs=np.concatenate(outs,axis=0)
                ssim=0
                for i in range(raw_outs.shape[0]):
                    ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
                ssim/=raw_outs.shape[0]
                # print(f"Block {blocki} method={attn.processor.steps_method} SSIM {ssim} target {ssim_thresholds[blocki]}" )
                print(cos_sim_threshold,ssim,ssim_thresholds[blocki])
                
                if ssim>ssim_thresholds[blocki]:
                    cos_sim_threshold_ed=cos_sim_threshold
                else:
                    cos_sim_threshold_st=cos_sim_threshold
            print(f"Block {blocki} calibration finished with ssim={ssim} \n{attn.processor.steps_method}")
        # save cache
        if not os.path.exists("cache"):
            os.makedirs("cache")
        torch.save(all_steps_method,cache_file)
    
    # set processor
    # for blocki, block in enumerate(blocks):
    #     attn: Attention = block.attn1
    #     attn.set_processor(FastAttnProcessor(window_size,all_steps_method[blocki]))
    
    # statistics
    counts=collections.Counter([method for block in blocks for method in block.attn1.processor.steps_method])
    # print(f"Counts {counts}")
    # compute fraction
    total=sum(counts.values())
    for k,v in counts.items():
        print(f"{k} {v/total}")

    # test final ssim
    outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
    outs=np.concatenate(outs,axis=0)
    ssim=0
    for i in range(raw_outs.shape[0]):
        ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
    ssim/=raw_outs.shape[0]
    print(f"Final SSIM {ssim}")

    return pipe,ssim
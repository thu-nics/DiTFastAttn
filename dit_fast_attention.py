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
import torch.nn as nn
import os
import time
from utils import set_profile_transformer_block_hook,process_profile_transformer_block

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
        residual = hidden_states
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
        attn.stepi += 1
        return hidden_states

class FastFeedForward(nn.Module):
    def __init__(self,net,steps_method):
        super().__init__()
        self.net=net
        self.steps_method=steps_method
        self.stepi=None
        self.cache_output=None
    
    def forward(self,hidden_states):
        out=hidden_states
        method=self.steps_method[self.stepi]
        if method=="output_share":
            out=self.cache_output
        elif "cfg_attn_share" in method:
            batch_size=hidden_states.shape[0]
            out=out[:batch_size//2]
            for module in self.net:
                out=module(out)
            out=torch.cat([out, out], dim=0)
            self.cache_output=out
        else:
            for module in self.net:
                out=module(out)
            self.cache_output=out
        self.stepi+=1
        return out

def set_stepi_warp(pipe):
    @functools.wraps(pipe)
    def wrapper(*args, **kwargs):
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            for layer in block.children():
                layer.stepi=0
                layer.cached_residual=None
                layer.cached_output=None
            # print(f"Reset stepi to 0 for block {blocki}")
        return pipe(*args, **kwargs)

    return wrapper

def eval_sensitivities(blocks,full_attn_macs,n_steps,pipe,calib_x,window_size,seed,raw_outs):
    sensitivities=[]
    for blocki, block in enumerate(blocks):
        attn=block.attn1
        sensitivities.append((blocki,0,"full_attn",full_attn_macs[blocki],1))
        for stepi in range(1,n_steps):
            for method in ["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]:
                methods=["full_attn" for _ in range(n_steps)]
                methods[stepi]=method
                
                attn.set_processor(FastAttnProcessor(window_size,methods))
                block.ff.steps_method=methods
                handler_collection=set_profile_transformer_block_hook(block)
                outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
                macs,_,info=process_profile_transformer_block(block,handler_collection,ret_layer_info=True)
                # breakpoint()
                macs=macs-full_attn_macs[blocki]*(n_steps-1)
                outs=np.concatenate(outs,axis=0)
                ssim=0
                for i in range(raw_outs.shape[0]):
                    ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
                ssim/=raw_outs.shape[0]
                # for stepi in range(1,n_steps):
                #     sensitivities.append((blocki,stepi,method,macs,ssim))
                sensitivities.append((blocki,stepi,method,macs,ssim))
                print(f"Block {blocki} step {stepi} method {method} SSIM {ssim} macs {macs}")
    return sensitivities

def eval_sensitivities_dist(blocks,full_attn_macs,n_steps,pipe,calib_x,window_size,seed,raw_outs):
    sensitivities=[]
    for blocki, block in enumerate(blocks):
        attn=block.attn1
        sensitivities.append((blocki,0,"full_attn",full_attn_macs[blocki],1))
        for stepi in range(1,n_steps):
            for method in ["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]:
                methods=["full_attn" for _ in range(n_steps)]
                methods[stepi]=method
                
                attn.set_processor(FastAttnProcessor(window_size,methods))
                block.ff.steps_method=methods
                handler_collection=set_profile_transformer_block_hook(block)
                outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
                macs,_,info=process_profile_transformer_block(block,handler_collection,ret_layer_info=True)
                # breakpoint()
                macs=macs-full_attn_macs[blocki]*(n_steps-1)
                outs=np.concatenate(outs,axis=0)
                ssim=0
                for i in range(raw_outs.shape[0]):
                    ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
                ssim/=raw_outs.shape[0]
                # for stepi in range(1,n_steps):
                #     sensitivities.append((blocki,stepi,method,macs,ssim))
                sensitivities.append((blocki,stepi,method,macs,ssim))
                print(f"Block {blocki} step {stepi} method {method} SSIM {ssim} macs {macs}")
    return sensitivities

def transform_model_fast_attention(raw_pipe, n_steps, n_calib, calib_x, threshold, window_size=[-64,64],use_cache=False,seed=3,sequential_calib=False,debug=False):
    pipe = set_stepi_warp(raw_pipe)
    blocks=pipe.transformer.transformer_blocks
    
    # evaluate sensitivity
    sensitivity_cache_file=f"cache/sensitivity_{raw_pipe.config._name_or_path.replace('/','_')}_{n_steps}_{n_calib}.pt"
    if use_cache and os.path.exists(sensitivity_cache_file):
        sensitivities=torch.load(sensitivity_cache_file)
    else:
        # reset all processors
        for blocki, block in enumerate(blocks):
            attn: Attention = block.attn1
            attn.raw_processor=attn.processor
            attn.set_processor(FastAttnProcessor(window_size,["full_attn" for _ in range(n_steps)]))
            block.ff=FastFeedForward(block.ff.net,attn.processor.steps_method)
        # evaluate macs for full attn
        full_attn_macs=[]
        raw_total_macs=0
        for blocki,block in enumerate(blocks):
            handler_collection=set_profile_transformer_block_hook(block)
            block.handler_collection=handler_collection
        raw_outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
        raw_outs=np.concatenate(raw_outs,axis=0)
        for blocki,block in enumerate(blocks):
            macs,_=process_profile_transformer_block(block,block.handler_collection)
            full_attn_macs.append(macs/n_steps)
            raw_total_macs+=macs
        
        sensitivities=eval_sensitivities(blocks,full_attn_macs,n_steps,pipe,calib_x,window_size,seed,raw_outs)
        
        
        torch.save(sensitivities,sensitivity_cache_file)
        # # test final ssim
        # outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
        # outs=np.concatenate(outs,axis=0)
        # ssim=0
        # for i in range(raw_outs.shape[0]):
        #     ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
        # ssim/=raw_outs.shape[0]
        # print(f"Final SSIM {ssim}")
    
    
    
    # binary search
    # sort sensitivities
    sorted_sensitivities=sorted(sensitivities,key=lambda x:x[-1])
    st=0
    ed=len(sensitivities)-1
    frac_target=threshold
    while ed-st>1:
        mid=(st+ed)//2
        blocks_methods={}
        blocks_macs={}
        for blocki,stepi,method,macs,ssim in sensitivities:
            if (blocki,stepi,method,macs,ssim) in sorted_sensitivities[:mid]:
                blocks_methods[(blocki,stepi)]="full_attn"
                blocks_macs[(blocki,stepi)]=full_attn_macs[blocki]
            else:
                blocks_methods[(blocki,stepi)]=method
                blocks_macs[(blocki,stepi)]=macs
        total_macs=sum(blocks_macs.values())
        frac=total_macs/raw_total_macs
        if frac>frac_target:
            ed=mid
        else:
            st=mid
    for blocki,block in enumerate(blocks):
        attn=block.attn1
        processor=FastAttnProcessor(window_size,[blocks_methods[(blocki,stepi)] for stepi in range(n_steps)])
        print(f"Block {blocki} method {processor.steps_method}")
        attn.set_processor(processor)
        block.ff=FastFeedForward(block.ff.net,attn.processor.steps_method)
    print(f"Final MAC Frac {frac}")
        
    # statistics
    counts=collections.Counter([method for block in blocks for method in block.attn1.processor.steps_method])
    # print(f"Counts {counts}")
    # compute fraction
    total=sum(counts.values())
    for k,v in counts.items():
        print(f"{k} {v/total}")

    

    return pipe,ssim
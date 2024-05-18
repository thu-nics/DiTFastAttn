import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
import functools
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import collections
import numpy as np
from modules.fast_feed_forward import FastFeedForward
from modules.fast_attn_processor import FastAttnProcessor
import os
import time
from utils import set_profile_transformer_block_hook,process_profile_transformer_block
import json

def set_stepi_warp(pipe):
    @functools.wraps(pipe)
    def wrapper(*args, **kwargs):
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            for layer in block.children():
                layer.stepi=0
                layer.cached_residual=None
                layer.cached_output=None
            # print(f"Reset stepi to 0 for block {blocki}")
        out=pipe(*args, **kwargs)
        for blocki, block in enumerate(pipe.transformer.transformer_blocks):
            for layer in block.children():
                layer.stepi=0
                layer.cached_residual=None
                layer.cached_output=None
        return out

    return wrapper

def mse_similarity(a,b):
    diff=(a-b)/(torch.max(a,b)+1e-6)
    return 1-diff.abs().clip(0,10).mean()

def sample_noise_output(noise_pred,pipe,latent_model_input,guidance_scale,t):
    latent_channels=pipe.transformer.config.in_channels
    # perform guidance
    if guidance_scale > 1:
        eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        noise_pred = torch.cat([eps, rest], dim=1)

    # learned sigma
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
    else:
        model_output = noise_pred

    # compute previous image: x_t -> x_t-1
    latent_model_input = pipe.scheduler.step(model_output, t, latent_model_input).prev_sample
    return latent_model_input

from diffusers.models.transformers.transformer_2d import Transformer2DModel
def transformer_forward_pre_hook(m:Transformer2DModel, args, kwargs):
    now_stepi=m.transformer_blocks[0].attn1.stepi
    for blocki,block in enumerate(m.transformer_blocks):
        block.attn1.processor.need_compute_residual[now_stepi]=False
        block.attn1.processor.need_cache_output=False
    raw_outs=m.forward(*args, **kwargs)[0]
    # raw_outs_im=sample_noise_output(raw_outs,m.pipe,args[0],4,kwargs["timestep"].flatten()[0])
    for blocki,block in enumerate(m.transformer_blocks):
        if now_stepi==0:
            continue
        # calibrate attn1
        for attni,attn in enumerate([block.attn1,block.attn2]):
            if attn is None or not isinstance(attn.processor, FastAttnProcessor):
                continue
            if attni==0:
                method_candidates=["full_attn+cfg_attn_share","residual_window_attn","residual_window_attn+cfg_attn_share","output_share"]
            else:
                method_candidates=["full_attn+cfg_attn_share","output_share"]
            selected_method="full_attn"
            for method in method_candidates:
                attn.processor.steps_method[now_stepi]=method
                # compute output
                for _block in m.transformer_blocks:
                    for layer in _block.children():
                        layer.stepi=now_stepi
                outs=m.forward(*args, **kwargs)[0]#.cpu().numpy()
                
                ssim=mse_similarity(raw_outs,outs)
                # outs_im=sample_noise_output(outs,m.pipe,args[0],4,kwargs["timestep"].flatten()[0])
                # bs=outs_im.shape[0]
                # ssim=mse_similarity(raw_outs_im[:bs//2],outs_im[:bs//2])
                # ssim=0
                # for i in range(raw_outs.shape[0]):
                #     ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
                # ssim/=raw_outs.shape[0]
                target=m.ssim_thresholds[now_stepi][blocki]
                # print(f">== {method}: {ssim}" )
                
                if ssim>target:
                    selected_method=method
            attn.processor.steps_method[now_stepi]=selected_method
            print(f"Block {blocki} attn{attni} stepi{now_stepi} {selected_method}")
        
        # calibrate ff
        if isinstance(block.ff, FastFeedForward):
            selected_method="full_attn"
            for method in ["cfg_attn_share","output_share"]:
                block.ff.steps_method[now_stepi]=method
                # compute output
                for _block in m.transformer_blocks:
                    for layer in _block.children():
                        layer.stepi=now_stepi
                outs=m.forward(*args, **kwargs)[0]#.cpu().numpy()
                ssim=mse_similarity(raw_outs,outs)
                # ssim=0
                # for i in range(raw_outs.shape[0]):
                #     ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=0, data_range=raw_outs.max() - raw_outs.min())
                # ssim/=raw_outs.shape[0]
                target=m.ssim_thresholds[now_stepi][blocki]
                # print(f"Block {blocki} step {now_stepi} method={method} SSIM {ssim} target {target}" )
                
                if ssim>target:
                    selected_method=method
            block.ff.steps_method[now_stepi]=selected_method
            print(f"Block {blocki} ff stepi{now_stepi} {selected_method}")


    for _block in m.transformer_blocks:
        for layer in _block.children():
            layer.stepi=now_stepi

    for blocki,block in enumerate(m.transformer_blocks):
        block.attn1.processor.need_compute_residual[now_stepi]=True
        block.attn1.processor.need_cache_output=True

@torch.no_grad()
def transform_model_fast_attention(raw_pipe, n_steps, n_calib, calib_x, threshold, window_size=[-64,64],use_cache=False,seed=3,sequential_calib=False,debug=False):
    pipe = set_stepi_warp(raw_pipe)
    blocks=pipe.transformer.transformer_blocks
    transformer:Transformer2DModel=pipe.transformer
    is_transform_attn2=blocks[0].attn2 is not None
    is_transform_attn2=False
    print(f"Transform attn2 {is_transform_attn2}")
    # is_transform_ff=hasattr(blocks[0],"ff")
    is_transform_ff=False
    print(f"Transform ff {is_transform_ff}")
    
    # calibration raw
    generator=torch.manual_seed(seed)
    raw_outs=pipe(calib_x,num_inference_steps=n_steps,generator=generator,output_type='np',return_dict=False)
    raw_outs=np.concatenate(raw_outs,axis=0)

    cache_file=f"cache/{raw_pipe.config._name_or_path.replace('/','_')}_{n_steps}_{n_calib}_{threshold}_{sequential_calib}.json"
    
    if use_cache and os.path.exists(cache_file):
        blocks_methods=torch.load(cache_file)
    else:
        # reset all processors
        for blocki, block in enumerate(blocks):
            attn: Attention = block.attn1
            block.attn1.set_processor(FastAttnProcessor(window_size,["full_attn" for _ in range(n_steps)]))
            block.attn1.processor.need_compute_residual=[True for _ in range(n_steps)]
            if is_transform_attn2:
                block.attn2.set_processor(FastAttnProcessor(window_size,["full_attn" for _ in range(n_steps)]))
                block.attn2.processor.need_compute_residual=[True for _ in range(n_steps)]
            if is_transform_ff:
                block.ff=FastFeedForward(block.ff.net,["full_attn" for _ in range(n_steps)])

        # ssim_theshold for each calibration
        ssim_thresholds=[]
        # all_steps=blocki*len(blocks)
        interval=(1-threshold)/len(blocks)
        for step_i in range(n_steps):
            sub_list=[]
            now_threshold=1
            for blocki in range(len(blocks)):
                now_threshold-=interval
                sub_list.append(now_threshold)
                # if sequential_calib:
                #     sub_list.append(1-interval*(blocki*n_steps+step_i))
                # else:
                #     sub_list.append(threshold)
            ssim_thresholds.append(sub_list)

        # calibration
        h=transformer.register_forward_pre_hook(transformer_forward_pre_hook,with_kwargs=True)
        transformer.ssim_thresholds=ssim_thresholds
        transformer.pipe=pipe

        outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
        outs=np.concatenate(outs,axis=0)
        ssim=0
        for i in range(raw_outs.shape[0]):
            ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
        ssim/=raw_outs.shape[0]
        print(f"Final SSIM {ssim}")

        h.remove()

        blocks_methods=[]
        for blocki, block in enumerate(blocks):
            attn_steps_method=block.attn1.processor.steps_method
            attn2_steps_method=block.attn2.processor.steps_method if is_transform_attn2 else None
            ff_steps_method=block.ff.steps_method if is_transform_ff else None
            blocks_methods.append({
                "attn1":attn_steps_method,
                "attn2":attn2_steps_method,
                "ff":ff_steps_method
            })

        # save cache
        if not os.path.exists("cache"):
            os.makedirs("cache")
        torch.save(blocks_methods,cache_file)
    
    # set processor
    for blocki, block in enumerate(blocks):
        block.attn1.set_processor(FastAttnProcessor(window_size,blocks_methods[blocki]["attn1"]))
        if blocks_methods[blocki]["attn2"] is not None:
            block.attn2.set_processor(FastAttnProcessor(window_size,blocks_methods[blocki]["attn2"]))
        if blocks_methods[blocki]["ff"] is not None:
            block.ff=FastFeedForward(block.ff.net,blocks_methods[blocki]["ff"])
    
    # statistics
    counts=collections.Counter([method for block in blocks for method in block.attn1.processor.steps_method])
    total=sum(counts.values())
    for k,v in counts.items():
        print(f"attn1 {k} {v/total}")

    if is_transform_attn2:
        counts=collections.Counter([method for block in blocks for method in block.attn2.processor.steps_method])
        total=sum(counts.values())
        for k,v in counts.items():
            print(f"attn2 {k} {v/total}")
    if is_transform_ff:
        counts=collections.Counter([method for block in blocks for method in block.ff.steps_method])
        total=sum(counts.values())
        for k,v in counts.items():
            print(f"ff {k} {v/total}")

    # test final ssim
    outs=pipe(calib_x,num_inference_steps=n_steps,generator=torch.manual_seed(seed),output_type='np',return_dict=False)
    outs=np.concatenate(outs,axis=0)
    ssim=0
    for i in range(raw_outs.shape[0]):
        ssim+=structural_similarity(raw_outs[i],outs[i], channel_axis=2, data_range=raw_outs.max() - raw_outs.min())
    ssim/=raw_outs.shape[0]
    print(f"Final SSIM {ssim}")
    # ssim=None

    return pipe,ssim
    
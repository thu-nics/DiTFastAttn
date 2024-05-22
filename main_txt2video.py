from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
import argparse
from evaluation import evaluate_quantitative_scores,evaluate_quantitative_scores_text2img,test_latencies
from dit_fast_attention import transform_model_fast_attention
import os
import json
import numpy as np
from utils import calculate_flops
from opensora.models.layers.blocks import Attention

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import IMG_FPS, save_sample
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from mmengine.config import Config
from utils import profile_pipe_transformer,count_flops_attn
from opensora_utils import *

def main():
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = cfg.prompt

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device="cpu")  # T5 must be fp32

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    from opensora.schedulers.iddpm import IDDPM
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # 4. inference
    # ======================================================
    save_dir = cfg.save_dir
    cfg.n_calib=cfg.scheduler.num_sampling_steps
    save_dir+=f"_{cfg.n_calib}_{cfg.n_steps}_{cfg.threshold}_{cfg.window_size}_{cfg.image_size}"
    os.makedirs(save_dir, exist_ok=True)
    pipe=OpensoraPipe(cfg,text_encoder,model,vae,scheduler,save_dir)
    # pipe(prompts)
    for blocki,block in enumerate(pipe.transformer.blocks):
        block.attn1=block.attn
    
    pipe.transformer.transformer_blocks=pipe.transformer.blocks
    from argparse import Namespace
    pipe.config=Namespace(_name_or_path="opensorav1.1")

    # macs, attn_mac=opensora_calculate_flops(pipe, prompts[:1])

    if cfg.threshold<1:
        pipe,ssim=transform_model_fast_attention(pipe, n_steps=cfg.n_steps, n_calib=cfg.n_calib, calib_x=prompts[:1], 
                                    threshold=cfg.threshold, window_size=[-cfg.window_size,cfg.window_size],
                                    use_cache=cfg.use_cache,seed=3, sequential_calib=cfg.sequential_calib,debug=cfg.debug,ablation=["residual_window_attn","output_share"])

    
    macs, attn_mac=opensora_calculate_flops(pipe, prompts[:1])
    
    with open("output/opensora_results.txt", "a+") as f:
        f.write(f"{cfg}\n{save_dir}\nmacs={macs}\nattn_mac={attn_mac}\n\n")
    
    set_random_seed(seed=cfg.seed)
    pipe(prompts)

    # pipe,calib_ssim=transform_model_fast_attention(pipe, n_steps=args.n_steps, n_calib=args.n_calib, calib_x=calib_x, threshold=args.threshold, window_size=[-args.window_size,args.window_size],use_cache=args.use_cache,seed=3, sequential_calib=args.sequential_calib,debug=args.debug)

    # macs, attn_mac=calculate_flops(pipe, calib_x[0:1],n_steps=args.n_steps)
    # latencies=test_latencies(pipe, args.n_steps,calib_x,bs=[1,])
    # if args.debug:
    #     result={}
    # else:
    #     result = evaluate_quantitative_scores_text2img(
    #         pipe, args.eval_real_image_path, mscoco_anno, args.eval_n_images, args.eval_batchsize,num_inference_steps=args.n_steps, fake_image_path=fake_image_path
    #     )
    # # save the result
    # print(result)
    


if __name__ == "__main__":
    main()

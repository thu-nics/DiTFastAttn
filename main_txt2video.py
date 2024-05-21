from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
import argparse
from evaluation import evaluate_quantitative_scores,evaluate_quantitative_scores_text2img,test_latencies
from dit_fast_attention import transform_model_fast_attention
import os
import json
import numpy as np
from utils import calculate_flops

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--sequential_calib", action="store_true")
    parser.add_argument("--eval_real_image_path", type=str, default="data/real_images")
    parser.add_argument("--coco_path", type=str, default="data/mscoco")
    parser.add_argument("--eval_n_images", type=int, default=5000)
    parser.add_argument("--eval_batchsize", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--raw_eval", action="store_true")
    args = parser.parse_args()
    
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
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32

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
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution == "PixArtMS":
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)
    elif cfg.multi_resolution == "STDiT2":
        image_size = cfg.image_size
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(cfg.batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        num_frames = torch.tensor([cfg.num_frames], device=device, dtype=dtype).repeat(cfg.batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        if cfg.num_frames == 1:
            cfg.fps = IMG_FPS
        fps = torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(cfg.batch_size)
        model_args["height"] = height
        model_args["width"] = width
        model_args["num_frames"] = num_frames
        model_args["ar"] = ar
        model_args["fps"] = fps

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    if cfg.sample_name is not None:
        sample_name = cfg.sample_name
    elif cfg.prompt_as_path:
        sample_name = ""
    else:
        sample_name = "sample"
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    for i in range(0, len(prompts), cfg.batch_size):
        # 4.2 sample in hidden space
        batch_prompts_raw = prompts[i : i + cfg.batch_size]
        batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts_raw]
        # handle the last batch
        if len(batch_prompts_raw) < cfg.batch_size and cfg.multi_resolution == "STDiT2":
            model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

        # 4.3. diffusion sampling
        old_sample_idx = sample_idx
        # generate multiple samples for each prompt
        for k in range(cfg.num_sample):
            sample_idx = old_sample_idx

            # Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if cfg.prompt_as_path:
                skip = True
                for batch_prompt in batch_prompts_raw:
                    path = os.path.join(save_dir, f"{sample_name}{batch_prompt}")
                    if cfg.num_sample != 1:
                        path = f"{path}-{k}"
                    path = f"{path}.mp4"
                    if not os.path.exists(path):
                        skip = False
                        break
                if skip:
                    continue

            # sampling
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
            )
            samples = vae.decode(samples.to(dtype))

            # 4.4. save samples
            if not use_dist or coordinator.is_master():
                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts_raw[idx]}")
                    if cfg.prompt_as_path:
                        sample_name_suffix = batch_prompts_raw[idx]
                    else:
                        sample_name_suffix = f"_{sample_idx}"
                    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
                    if cfg.num_sample != 1:
                        save_path = f"{save_path}-{k}"
                    save_sample(sample, fps=cfg.fps // cfg.frame_interval, save_path=save_path)
                    sample_idx += 1
    
    
    if args.model in ["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS","PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"]:
        from diffusers import Transformer2DModel, PixArtSigmaPipeline
        transformer = Transformer2DModel.from_pretrained(
            args.model,
            subfolder='transformer', 
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        pipe = PixArtSigmaPipeline.from_pretrained(
            'PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers',
            transformer=transformer,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.config._name_or_path=args.model

        
        pipe.to("cuda")
    else:
        raise NotImplementedError

    with open(f"{args.coco_path}/annotations/captions_val2017.json") as f:
        mscoco_anno = json.load(f)
    
    # set seed
    np.random.seed(3)
    slice=np.random.choice(mscoco_anno['annotations'],args.n_calib)
    calib_x = [d['caption'] for d in slice]
    
    if args.raw_eval:
        fake_image_path = f"output/{args.model.replace('/','_')}_steps{args.n_steps}"
        calib_ssim=1
    else:
        
        pipe,calib_ssim=transform_model_fast_attention(pipe, n_steps=args.n_steps, n_calib=args.n_calib, calib_x=calib_x, threshold=args.threshold, window_size=[-args.window_size,args.window_size],use_cache=args.use_cache,seed=3, sequential_calib=args.sequential_calib,debug=args.debug)

        fake_image_path = f"output/{args.model.replace('/','_')}_calib{args.n_calib}_steps{args.n_steps}_threshold{args.threshold}_window{args.window_size}_seq{args.sequential_calib}"
        
    
    macs, attn_mac=calculate_flops(pipe, calib_x[0:1],n_steps=args.n_steps)
    latencies=test_latencies(pipe, args.n_steps,calib_x,bs=[1,])
    if args.debug:
        result={}
    else:
        result = evaluate_quantitative_scores_text2img(
            pipe, args.eval_real_image_path, mscoco_anno, args.eval_n_images, args.eval_batchsize,num_inference_steps=args.n_steps, fake_image_path=fake_image_path
        )
    # save the result
    print(result)
    with open("output/results.txt", "a+") as f:
        f.write(f"{args}\ncalib_ssim={calib_ssim}\n{result}\nmacs={macs}\nattn_mac={attn_mac}\nlatencies={latencies}\n\n")


if __name__ == "__main__":
    main()

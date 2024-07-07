from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
import argparse
from evaluation import (
    evaluate_quantitative_scores,
    evaluate_quantitative_scores_text2img,
    test_latencies,
)
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
from utils import profile_pipe_transformer, count_flops_attn


def opensora_calculate_flops(pipe, x):
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            module.heads = module.num_heads
    macs, params, attn_ops, attn2_ops = profile_pipe_transformer(
        pipe,
        inputs=(x,),
        kwargs={},
        custom_ops={Attention: count_flops_attn},
        verbose=0,
        ret_layer_info=True,
    )
    print(
        f"macs is {macs/1e9} G, attn is {(attn_ops)/1e9} G, attn2_ops is {(attn2_ops)/1e9} G"
    )
    return macs / 1e9, attn_ops / 1e9


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument(
        "--config", default="data/sample.py", help="model config file path"
    )

    # DiTFastAttn
    parser.add_argument("--n_calib", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--sequential_calib", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--raw_eval", action="store_true")

    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="path to model ckpt; will overwrite cfg.ckpt_path if specified",
    )
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")

    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument(
            "--save-dir", default=None, type=str, help="path to save generated samples"
        )
        parser.add_argument(
            "--sample-name",
            default=None,
            type=str,
            help="sample name, default is sample_idx",
        )
        parser.add_argument(
            "--start-index", default=None, type=int, help="start index for sample name"
        )
        parser.add_argument(
            "--end-index", default=None, type=int, help="end index for sample name"
        )
        parser.add_argument(
            "--num-sample",
            default=None,
            type=int,
            help="number of samples to generate for one prompt",
        )
        parser.add_argument(
            "--prompt-as-path",
            action="store_true",
            help="use prompt as path to save samples",
        )

        # prompt
        parser.add_argument(
            "--prompt-path", default=None, type=str, help="path to prompt txt file"
        )
        parser.add_argument(
            "--prompt", default=None, type=str, nargs="+", help="prompt list"
        )

        # image/video
        parser.add_argument(
            "--num-frames", default=None, type=int, help="number of frames"
        )
        parser.add_argument("--fps", default=None, type=int, help="fps")
        parser.add_argument(
            "--image-size", default=None, type=int, nargs=2, help="image size"
        )

        # hyperparameters
        parser.add_argument(
            "--num-sampling-steps", default=None, type=int, help="sampling steps"
        )
        parser.add_argument(
            "--cfg-scale",
            default=None,
            type=float,
            help="balance between cond & uncond",
        )

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument(
            "--condition-frame-length",
            default=None,
            type=int,
            help="condition frame length",
        )
        parser.add_argument(
            "--reference-path", default=None, type=str, nargs="+", help="reference path"
        )
        parser.add_argument(
            "--mask-strategy", default=None, type=str, nargs="+", help="mask strategy"
        )
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument(
            "--load", default=None, type=str, help="path to continue training"
        )
        parser.add_argument(
            "--data-path", default=None, type=str, help="path to data csv"
        )
        parser.add_argument(
            "--start-from-scratch",
            action="store_true",
            help="start training from scratch",
        )

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if training and args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    if not training and args.cfg_scale is not None:
        cfg.scheduler["cfg_scale"] = args.cfg_scale
        args.cfg_scale = None
    if not training and args.num_sampling_steps is not None:
        cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
        args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    if not training:
        # Inference only
        # - Allow not set
        if "reference_path" not in cfg:
            cfg["reference_path"] = None
        if "loop" not in cfg:
            cfg["loop"] = 1
        if "frame_interval" not in cfg:
            cfg["frame_interval"] = 1
        if "sample_name" not in cfg:
            cfg["sample_name"] = None
        if "num_sample" not in cfg:
            cfg["num_sample"] = 1
        if "prompt_as_path" not in cfg:
            cfg["prompt_as_path"] = False
        # - Prompt handling
        if "prompt" not in cfg or cfg["prompt"] is None:
            assert (
                cfg["prompt_path"] is not None
            ), "prompt or prompt_path must be provided"

            def load_prompts(prompt_path):
                with open(prompt_path, "r") as f:
                    prompts = [line.strip() for line in f.readlines()]
                return prompts

            cfg["prompt"] = load_prompts(cfg["prompt_path"])
        if args.start_index is not None and args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index : args.end_index]
        elif args.start_index is not None:
            cfg["prompt"] = cfg["prompt"][args.start_index :]
        elif args.end_index is not None:
            cfg["prompt"] = cfg["prompt"][: args.end_index]
    else:
        # Training only
        # - Allow not set
        if "mask_ratios" not in cfg:
            cfg["mask_ratios"] = None
        if "start_from_scratch" not in cfg:
            cfg["start_from_scratch"] = False
        if "bucket_config" not in cfg:
            cfg["bucket_config"] = None
        if "transform_name" not in cfg.dataset:
            cfg.dataset["transform_name"] = "center"
        if "num_bucket_build_workers" not in cfg:
            cfg["num_bucket_build_workers"] = 1

    # Both training and inference
    if "multi_resolution" not in cfg:
        cfg["multi_resolution"] = False

    return cfg


def parse_configs(training=False):
    args = parse_args(training)
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args, training)
    return cfg


class OpensoraPipe:
    def __init__(self, cfg, text_encoder, transformer, vae, scheduler, save_dir):
        self.cfg = cfg
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.save_dir = save_dir

    def __call__(self, prompts, **kwargs):
        cfg = self.cfg
        vae = self.vae
        scheduler = self.scheduler
        device = "cuda"
        dtype = torch.bfloat16
        model_args = dict()
        if cfg.multi_resolution == "PixArtMS":
            image_size = cfg.image_size
            hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(
                cfg.batch_size, 1
            )
            ar = torch.tensor(
                [[image_size[0] / image_size[1]]], device=device, dtype=dtype
            ).repeat(cfg.batch_size, 1)
            model_args["data_info"] = dict(ar=ar, hw=hw)
        elif cfg.multi_resolution == "STDiT2":
            image_size = cfg.image_size
            height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(
                cfg.batch_size
            )
            width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(
                cfg.batch_size
            )
            num_frames = torch.tensor(
                [cfg.num_frames], device=device, dtype=dtype
            ).repeat(cfg.batch_size)
            ar = torch.tensor(
                [image_size[0] / image_size[1]], device=device, dtype=dtype
            ).repeat(cfg.batch_size)
            if cfg.num_frames == 1:
                cfg.fps = IMG_FPS
            fps = torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(
                cfg.batch_size
            )
            model_args["height"] = height
            model_args["width"] = width
            model_args["num_frames"] = num_frames
            model_args["ar"] = ar
            model_args["fps"] = fps
        input_size = (cfg.num_frames, *cfg.image_size)
        latent_size = vae.get_latent_size(input_size)

        # 4.1. batch generation
        sample_idx = 0
        for i in range(0, len(prompts), cfg.batch_size):
            # 4.2 sample in hidden space
            batch_prompts_raw = prompts[i : i + cfg.batch_size]
            batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts_raw]

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
                        path = os.path.join(self.save_dir, f"{batch_prompt}")
                        if cfg.num_sample != 1:
                            path = f"{path}-{k}"
                        path = f"{path}.mp4"
                        if not os.path.exists(path):
                            skip = False
                            break
                    if skip:
                        continue

                # sampling

                for blocki, block in enumerate(self.transformer.transformer_blocks):
                    for layer in block.children():
                        layer.stepi = 0
                        layer.cached_residual = None
                        layer.cached_output = None

                z = torch.randn(
                    len(batch_prompts),
                    vae.out_channels,
                    *latent_size,
                    device=device,
                    dtype=dtype,
                )
                samples = scheduler.sample(
                    self.transformer,
                    self.text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    device=device,
                    additional_args=model_args,
                )
                samples = vae.decode(samples.to(dtype))

                # 4.4. save samples
                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts_raw[idx]}")
                    if cfg.prompt_as_path:
                        sample_name_suffix = batch_prompts_raw[idx]
                    else:
                        sample_name_suffix = f"_{sample_idx}"
                    save_path = os.path.join(self.save_dir, f"{sample_name_suffix}")
                    if cfg.num_sample != 1:
                        save_path = f"{save_path}-{k}"
                    save_sample(
                        sample, fps=cfg.fps // cfg.frame_interval, save_path=save_path
                    )
                    sample_idx += 1
        return sample.transpose(0, 1).float().cpu().numpy()  # C, T, H, W

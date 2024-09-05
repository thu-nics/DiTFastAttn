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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--sequential_calib", action="store_true")
    parser.add_argument("--eval_real_image_path", type=str, default="data/real_images_coco_30k")
    parser.add_argument("--coco_path", type=str, default="data/mscoco")
    parser.add_argument("--eval_n_images", type=int, default=30000)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--raw_eval", action="store_true")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    args = parser.parse_args()

    if args.model in [
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
    ]:
        from diffusers import Transformer2DModel, PixArtSigmaPipeline

        transformer = Transformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.config._name_or_path = args.model

        pipe.to("cuda")
    else:
        raise NotImplementedError

    with open(f"{args.coco_path}/annotations/captions_val2014.json") as f:
        mscoco_anno = json.load(f)

    # set seed
    np.random.seed(args.seed)
    slice_ = np.random.choice(mscoco_anno["annotations"], args.n_calib)
    calib_x = [d["caption"] for d in slice_]

    if args.raw_eval:
        fake_image_path = f"output/{args.model.replace('/','_')}_steps{args.n_steps}"
    else:

        pipe, search_time = transform_model_fast_attention(
            pipe,
            n_steps=args.n_steps,
            n_calib=args.n_calib,
            calib_x=calib_x,
            threshold=args.threshold,
            window_size=[args.window_size, args.window_size],
            use_cache=args.use_cache,
            seed=args.seed,
            sequential_calib=args.sequential_calib,
            debug=args.debug,
            cond_first=False,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.cfg_scale,
        )

        fake_image_path = f"output/{args.model.replace('/','_')}_calib{args.n_calib}_steps{args.n_steps}_threshold{args.threshold}_window{args.window_size}_seq{args.sequential_calib}_evalsize{args.eval_n_images}_cfg_scale{args.cfg_scale}"
        if args.metric != "":
            fake_image_path = fake_image_path + f"_metric{args.metric}"
        if args.negative_prompt != "":
            fake_image_path = fake_image_path + f"_negative_prompt{args.negative_prompt}"

    macs, attn_mac = calculate_flops(pipe, calib_x[0:1], n_steps=args.n_steps)
    latencies = test_latencies(
        pipe,
        args.n_steps,
        calib_x,
        bs=[
            1,
        ],
    )
    memory_allocated = torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024**2)
    memory_cached = torch.cuda.max_memory_cached(device=torch.device("cuda")) / (1024**2)
    if args.debug:
        result = {}
    else:
        result = evaluate_quantitative_scores_text2img(
            pipe,
            args.eval_real_image_path,
            mscoco_anno,
            args.eval_n_images,
            args.eval_batchsize,
            num_inference_steps=args.n_steps,
            fake_image_path=fake_image_path,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            guidance_scale=args.cfg_scale,
        )
    # save the result
    print(result)
    with open("output/results.txt", "a+") as f:
        f.write(
            f"{args}\n{result}\nmacs={macs}\nattn_mac={attn_mac}\nlatencies={latencies}\nmemory allocated={memory_allocated}MB\nmemory cached={memory_cached}MB\nsearch time={search_time}s\n\n"
        )


if __name__ == "__main__":
    main()

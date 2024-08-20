from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
import argparse
from evaluation import evaluate_quantitative_scores, test_latencies
from dit_fast_attention import transform_model_fast_attention
import os
from utils import calculate_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-512")
    parser.add_argument("--n_calib", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--sequential_calib", action="store_true")
    parser.add_argument("--eval_real_image_path", type=str, default="data/real_images_imagenet_50k")
    parser.add_argument("--eval_n_images", type=int, default=50000)
    parser.add_argument("--eval_batchsize", type=int, default=12)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--raw_eval", action="store_true")
    parser.add_argument("--ablation", type=str, default="")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4)
    args = parser.parse_args()

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    calib_x = torch.randint(0, 1000, (args.n_calib,), generator=torch.Generator().manual_seed(args.seed)).to("cuda")
    if args.raw_eval:
        fake_image_path = f"output/{args.model.replace('/','_')}_steps{args.n_steps}"
    else:
        pipe, search_time = transform_model_fast_attention(
            pipe,
            n_steps=args.n_steps,
            n_calib=args.n_calib,
            calib_x=calib_x,
            threshold=args.threshold,
            window_size=[-args.window_size, args.window_size],
            use_cache=args.use_cache,
            seed=3,
            sequential_calib=args.sequential_calib,
            debug=args.debug,
            ablation=args.ablation,
            cond_first=True,
            metric=args.metric,
            guidance_scale=args.cfg_scale,
        )
        # evaluate the results
        fake_image_path = f"output/{args.model.replace('/','_')}_calib{args.n_calib}_steps{args.n_steps}_threshold{args.threshold}_window{args.window_size}_sequential{args.sequential_calib}_evalsize{args.eval_n_images}_cfg_scale{args.cfg_scale}"
        if args.ablation != "":
            fake_image_path = fake_image_path + f"_ablation{args.ablation}"
        if args.metric != "":
            fake_image_path = fake_image_path + f"_{args.metric}"

    macs, attn_mac = calculate_flops(pipe, calib_x[0:1], n_steps=args.n_steps)
    latencies = test_latencies(pipe, args.n_steps, calib_x, bs=[8])
    memory_allocated = torch.cuda.max_memory_allocated(device=torch.device("cuda")) / (1024**2)
    memory_cached = torch.cuda.max_memory_cached(device=torch.device("cuda")) / (1024**2)
    if args.debug:
        result = {}
    else:
        result = evaluate_quantitative_scores(
            pipe,
            args.eval_real_image_path,
            args.eval_n_images,
            args.eval_batchsize,
            num_inference_steps=args.n_steps,
            fake_image_path=fake_image_path,
            seed=args.seed,
            guidance_scale=args.cfg_scale,
        )

    # save the result
    with open("output/results.txt", "a+") as f:
        f.write(
            f"{args}\n{result}\nmacs={macs}\nattn_mac={attn_mac}\nlatencies={latencies}\nmemory allocated={memory_allocated}MB\nmemory cached={memory_cached}MB\nsearch time={search_time}s\n\n"
        )


if __name__ == "__main__":
    main()

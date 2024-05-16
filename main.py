from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
import argparse
from evaluation import evaluate_quantitative_scores,evaluate_latencies
from dit_fast_attention import transform_model_fast_attention,speed_test
import os
from utils import calculate_flops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-512")
    parser.add_argument("--n_calib", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--sequential_calib", action="store_true")
    parser.add_argument("--eval_real_image_path", type=str, default="data/real_images")
    parser.add_argument("--eval_n_images", type=int, default=5000)
    parser.add_argument("--eval_batchsize", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--raw_eval", action="store_true")
    args = parser.parse_args()

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    calib_x=torch.randint(0, 1000, (args.n_calib,),generator=torch.Generator().manual_seed(3)).to("cuda")
    if args.raw_eval:
        fake_image_path = f"output/{args.model.replace('/','_')}_steps{args.n_steps}"
        calib_ssim=1
    else:
        pipe,calib_ssim=transform_model_fast_attention(pipe, n_steps=args.n_steps, n_calib=args.n_calib, calib_x=calib_x, threshold=args.threshold, window_size=[-args.window_size,args.window_size],use_cache=args.use_cache,seed=3, sequential_calib=args.sequential_calib,debug=args.debug)
        # evaluate the results
        fake_image_path = f"output/{args.model.replace('/','_')}_calib{args.n_calib}_steps{args.n_steps}_threshold{args.threshold}_window{args.window_size}_sequential{args.sequential_calib}"
        
    macs, attn_mac=calculate_flops(pipe, calib_x[0:1],n_steps=args.n_steps)
    if not args.debug:
        result = evaluate_quantitative_scores(
            pipe, args.eval_real_image_path, args.eval_n_images, args.eval_batchsize,num_inference_steps=args.n_steps, fake_image_path=fake_image_path
        )
    else:
        result={}
    latencies=evaluate_latencies(pipe, args.n_steps,calib_x,bs=[1,4,16])

    # save the result
    with open("output/results.txt", "a+") as f:
        f.write(f"{args}\ncalib_ssim={calib_ssim}\n{result}\nmacs={macs}\nattn_mac={attn_mac}\nlatencies={latencies}\n\n")


if __name__ == "__main__":
    main()

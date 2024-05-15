from diffusers import DiTPipeline, DPMSolverMultistepScheduler
import torch
import argparse
from evaluation import evaluate_quantitative_scores
from dit_fast_attention import transform_model_fast_attention
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-512")
    parser.add_argument("--n_calib", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=0.98)
    parser.add_argument("--eval_real_image_path", type=str, default="data/real_images")
    parser.add_argument("--eval_n_images", type=int, default=50)
    parser.add_argument("--eval_batchsize", type=int, default=25)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--nshare", type=int, default=2)
    args = parser.parse_args()

    raw_pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")

    # calib_dataset = [torch.randint(0, 1000, (args.n_calib_data,))]
    pipe=transform_model_fast_attention(raw_pipe, n_steps=args.n_steps, n_calib=args.n_calib, threshold=args.threshold, window_size=[-64,64],use_cache=False,seed=3)

    result = evaluate_quantitative_scores(
        pipe, args.eval_real_image_path, args.eval_n_images, args.eval_batchsize,num_inference_steps=args.n_steps
    )


if __name__ == "__main__":
    main()

CUDA_VISIBLE_DEVICES=5 python main_text2img.py --n_calib 8 --n_steps 20 --threshold 0.98 & 
CUDA_VISIBLE_DEVICES=6 python main_text2img.py --n_calib 4 --n_steps 20 --threshold 0.96

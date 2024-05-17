CUDA_VISIBLE_DEVICES=4 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.95 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.9 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.85 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.8 --use_cache --sequential

CUDA_VISIBLE_DEVICES=4 python main_txt2img.py --n_calib 4 --n_steps 20 --threshold 0.98 --use_cache & 
CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --n_calib 4 --n_steps 20 --threshold 0.95 --use_cache &
CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --n_calib 4 --n_steps 20 --threshold 0.92 --use_cache &
CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --n_calib 4 --n_steps 20 --threshold 0.88 --use_cache

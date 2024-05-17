# CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --n_calib 4 --n_steps 20 --window_size 256 --threshold 0.98 --use_cache &
CUDA_VISIBLE_DEVICES=4 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.9 --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.8 --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.7 --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.6 --eval_n_images 500

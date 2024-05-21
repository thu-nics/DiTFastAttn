CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.975 --eval_n_images 5000 --use_cache --debug &
CUDA_VISIBLE_DEVICES=1 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.95 --eval_n_images 5000 --use_cache --debug &
CUDA_VISIBLE_DEVICES=2 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.925 --eval_n_images 5000 --use_cache --debug &
CUDA_VISIBLE_DEVICES=3 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.9 --eval_n_images 5000 --use_cache --debug &
CUDA_VISIBLE_DEVICES=4 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.875 --eval_n_images 5000 --use_cache --debug &
CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.85 --eval_n_images 5000 --use_cache --debug &
# CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.825 --eval_n_images 5000 --use_cache &
# CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --n_calib 6 --n_steps 20 --window_size 512 --threshold 0.8 --eval_n_images 5000 --use_cache

# CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --threshold 1 --raw --debug


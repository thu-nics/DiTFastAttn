CUDA_VISIBLE_DEVICES=1 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.0 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=2 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.025 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=3 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.05 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=4 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.075 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=5 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.1 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=6 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.125 --eval_n_images 50000 --use_cache &
CUDA_VISIBLE_DEVICES=7 python run_dit.py --n_calib 8 --n_steps 50 --window_size 128 --threshold 0.15 --eval_n_images 50000 --use_cache &
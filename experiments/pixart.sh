CUDA_VISIBLE_DEVICES=0 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=1 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.025 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=2 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.05 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=3 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.075 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=4 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.1 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=5 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.125 --eval_n_images 30000 --use_cache &
CUDA_VISIBLE_DEVICES=6 python run_pixart.py --n_calib 6 --n_steps 50 --window_size 512 --threshold 0.15 --eval_n_images 30000 --use_cache &
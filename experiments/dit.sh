CUDA_VISIBLE_DEVICES=0 python main.py --n_calib 8 --n_steps 20 --threshold 0.98 --use_cache &
CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 8 --n_steps 20 --threshold 0.92 
# CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 4 --n_steps 20 --threshold 0.95 &
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 16 --n_steps 20 --threshold 0.95 

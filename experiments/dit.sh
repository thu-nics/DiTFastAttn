CUDA_VISIBLE_DEVICES=0 python main.py --n_calib 8 --n_steps 20 --threshold 0.95 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=1 python main.py --n_calib 8 --n_steps 20 --threshold 0.9 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=2 python main.py --n_calib 8 --n_steps 20 --threshold 0.85 --use_cache --sequential &
CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 8 --n_steps 20 --threshold 0.8 --use_cache --sequential
# CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 4 --n_steps 20 --threshold 0.95 &
# CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 16 --n_steps 20 --threshold 0.95 

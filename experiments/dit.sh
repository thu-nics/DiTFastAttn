CUDA_VISIBLE_DEVICES=2 python main.py --n_calib 8 --n_steps 20 --threshold 0.98 & 
CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 8 --n_steps 20 --threshold 0.95 &
CUDA_VISIBLE_DEVICES=4 python main.py --n_calib 8 --n_steps 20 --threshold 0.98 --independent_calib &
CUDA_VISIBLE_DEVICES=5 python main.py --n_calib 16 --n_steps 20 --threshold 0.95 &
CUDA_VISIBLE_DEVICES=6 python main.py --n_calib 32 --n_steps 20 --threshold 0.95

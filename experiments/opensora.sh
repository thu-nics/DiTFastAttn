CUDA_VISIBLE_DEVICES=0 python run_opensora.py --threshold 0.0 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=1 python run_opensora.py --threshold 0.01 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=2 python run_opensora.py --threshold 0.02 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=3 python run_opensora.py --threshold 0.03 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=4 python run_opensora.py --threshold 0.04 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=5 python run_opensora.py --threshold 0.05 --window_size 50 --n_calib 4 --use_cache &
CUDA_VISIBLE_DEVICES=6 python run_opensora.py --threshold 0.06 --window_size 50 --n_calib 4 --use_cache &
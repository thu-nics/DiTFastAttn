CUDA_VISIBLE_DEVICES=1 python run_opensora.py --threshold 0 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=2 python run_opensora.py --threshold 0.025 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=3 python run_opensora.py --threshold 0.05 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=4 python run_opensora.py --threshold 0.0725 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=5 python run_opensora.py --threshold 0.1 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=6 python run_opensora.py --threshold 0.125 --window_size 200 --image-size 480 854 --use_cache &
CUDA_VISIBLE_DEVICES=7 python run_opensora.py --threshold 0.15 --window_size 200 --image-size 480 854 --use_cache &
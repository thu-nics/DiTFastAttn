CUDA_VISIBLE_DEVICES=1 python main_txt2video.py --threshold 1 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=2 python main_txt2video.py --threshold 0.875 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=3 python main_txt2video.py --threshold 0.85 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=4 python main_txt2video.py --threshold 0.975 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=5 python main_txt2video.py --threshold 0.95 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=6 python main_txt2video.py --threshold 0.925 --window_size 32 --use_cache &
CUDA_VISIBLE_DEVICES=7 python main_txt2video.py --threshold 0.9 --window_size 32 --use_cache &

# CUDA_VISIBLE_DEVICES=0 python main_txt2video.py --threshold 0.95 --window_size 64 --image-size 480 854



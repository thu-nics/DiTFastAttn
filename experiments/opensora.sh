CUDA_VISIBLE_DEVICES=0 python main_txt2video.py --threshold 1 --use_cache &
CUDA_VISIBLE_DEVICES=0 python main_txt2video.py --threshold 0.975 --use_cache &
CUDA_VISIBLE_DEVICES=1 python main_txt2video.py --threshold 0.95 --use_cache &
CUDA_VISIBLE_DEVICES=2 python main_txt2video.py --threshold 0.925 --use_cache &
CUDA_VISIBLE_DEVICES=3 python main_txt2video.py --threshold 0.9 --use_cache 
# CUDA_VISIBLE_DEVICES=0 python main_txt2video.py --threshold 0.975 --use_cache  --image-size 480 854 &
# CUDA_VISIBLE_DEVICES=1 python main_txt2video.py --threshold 0.95 --use_cache  --image-size 480 854 &
# CUDA_VISIBLE_DEVICES=2 python main_txt2video.py --threshold 0.925 --use_cache  --image-size 480 854 &
# CUDA_VISIBLE_DEVICES=3 python main_txt2video.py --threshold 0.9 --use_cache  --image-size 480 854

# raw
# CUDA_VISIBLE_DEVICES=2 python scripts/inference.py ../../data/sample.py  --image-size 720 1280 --save-dir output/sample_720_1280/
# CUDA_VISIBLE_DEVICES=3 python scripts/inference.py ../../data/sample.py  --image-size 480 854 --save-dir output/sample_480_854/
# CUDA_VISIBLE_DEVICES=3 python scripts/inference.py ../../data/sample.py --save-dir output/sample/
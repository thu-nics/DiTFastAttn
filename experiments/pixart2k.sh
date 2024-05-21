CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.9 --window_size 2048  --eval_n_images 500 --use_cache&
CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.875 --window_size 2048  --eval_n_images 500 --use_cache &
CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.85 --window_size 2048  --eval_n_images 500 --use_cache &
# CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.8 --window_size 2048  --eval_n_images 5000 --use_cache 

CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.85 --window_size 2048  --eval_n_images 500 --use_cache --raw

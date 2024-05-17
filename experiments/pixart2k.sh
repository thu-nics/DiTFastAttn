CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.9 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=1 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.8 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=2 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.7 --window_size 512  --eval_n_images 500 &
CUDA_VISIBLE_DEVICES=3 python main_txt2img.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.6 --window_size 512  --eval_n_images 500 &

# CUDA_VISIBLE_DEVICES=0 python main_txt2img.py --n_calib 4 --n_steps 20 --window_size 256 --threshold 0.98 --use_cache &
CUDA_VISIBLE_DEVICES=4 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.95 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=5 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.85 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=6 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.75 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=7 python main_txt2img.py --n_calib 8 --n_steps 20 --window_size 256 --threshold 0.65 --eval_n_images 5000

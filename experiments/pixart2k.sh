CUDA_VISIBLE_DEVICES=0 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.025 --window_size 2048  --eval_n_images 5000 --use_cache &
CUDA_VISIBLE_DEVICES=1 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.05 --window_size 2048  --eval_n_images 5000 --use_cache &
CUDA_VISIBLE_DEVICES=2 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.0725 --window_size 2048  --eval_n_images 5000 --use_cache &
CUDA_VISIBLE_DEVICES=3 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.1 --window_size 2048  --eval_n_images 5000 --use_cache &
CUDA_VISIBLE_DEVICES=4 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.125 --window_size 2048  --eval_n_images 5000 --use_cache &
CUDA_VISIBLE_DEVICES=5 python run_pixart.py --model PixArt-alpha/PixArt-Sigma-XL-2-2K-MS --n_calib 4 --threshold 0.15 --window_size 2048  --eval_n_images 5000 --use_cache &


CUDA_VISIBLE_DEVICES=0 python main.py --n_calib 8 --n_steps 20 --threshold 0.975 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=1 python main.py --n_calib 8 --n_steps 20 --threshold 0.95 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=2 python main.py --n_calib 8 --n_steps 20 --threshold 0.925 --eval_n_images 5000 &
CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 8 --n_steps 20 --threshold 0.9 --eval_n_images 5000

# CUDA_VISIBLE_DEVICES=3 python main.py --n_calib 2 --n_steps 20 --threshold 0.99 --eval_n_images 5000

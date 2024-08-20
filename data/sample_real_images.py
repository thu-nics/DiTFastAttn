import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("image_folder", type=str)
parser.add_argument("--output_folder", type=str, default="data/real_images")
parser.add_argument("--n_images", type=int, default=30000)
args = parser.parse_args()

all_images = []
for root, _, files in os.walk(args.image_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            all_images.append(os.path.join(root, file))

sampled_images = np.random.choice(all_images, args.n_images, replace=False)
os.makedirs(args.output_folder, exist_ok=True)
for i, image in enumerate(sampled_images):
    os.system(f"cp {image} {args.output_folder}/{i}.jpg")
    if i % 100 == 0:
        print(f"processed {i} images")

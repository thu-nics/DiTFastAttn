from torchvision.transforms import functional as F
import torch.nn.functional
import torch
import numpy as np
from PIL import Image
import os
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from diffusers.models.attention_processor import Attention

def evaluate_template_matching(order_list, cal_cost, pipe):
    count = {0:0}
    for pattern_type in order_list:
        count[pattern_type] = 0
    mask_count_total = 0
    total = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            for timestep, mask_list in module.mask.items():
                pattern_list = np.zeros(16)
                for i in range(16):
                    type = 0
                    for j in order_list:
                        if mask_list[j][i]:
                             type = j
                             pattern_list[i] = j
                             break
                    count[type] += 1
                module.mask[timestep] = pattern_list
    
    total_num = sum(count.values())
    cal = 0
    for k,v in count.items():
        cal += cal_cost[k]* v / total_num
    print("template matching info: ")
    print(count)
    print("total percentage reduction: ", round(1 - cal, 2))
    


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


def evaluate_quantitative_scores(dit_pipeline,real_image_path,n_images=5000,batchsize=1,seed=3,num_inference_steps=50,fake_image_path="output/fake_images"):
    results={}
    # Inception Score
    inception = InceptionScore()
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path):
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for i in range(0, n_images, batchsize):
        class_ids = np.random.randint(0, 1000, batchsize)
        output = dit_pipeline(class_labels=class_ids, generator=generator, output_type="np",num_inference_steps=num_inference_steps)
        fake_images = output.images
        # Inception Score
        torch_images=torch.Tensor(fake_images*255).byte().permute(0,3,1,2).contiguous()
        torch_images=torch.nn.functional.interpolate(torch_images, size=(299, 299), mode='bilinear', align_corners=False)
        inception.update(torch_images)
        # save torch_image
        # for j, image in enumerate(torch_images):
        #     image = F.to_pil_image(image)
        #     image.save(f"{fake_image_path}/{i+j}_2.png")
        # save
        for j, image in enumerate(fake_images):
            image = F.to_pil_image(image)
            image.save(f"{fake_image_path}/{i+j}.png")
    
    IS=inception.compute()
    results["IS"]=IS
    print(f"Inception Score: {IS}")
    
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    fid_value=calculate_fid_given_paths([real_image_path,fake_image_path],64,device,dims=2048,num_workers=8)
    results["FID"]=fid_value
    print(f"FID: {fid_value}")
    
    
    
    return results
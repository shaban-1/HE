import warnings
import os
import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils import get_config
from trainer import UNIT_Trainer
from models.LUM_model import DecomNet

parser = argparse.ArgumentParser()
parser.add_argument('--denoise_config', type=str, default='./configs/unit_NDM.yaml')
parser.add_argument('--light_config', type=str, default='configs/unit_LUM.yaml')
parser.add_argument('--input_folder', type=str, default='./test_images')
parser.add_argument('--output_folder', type=str, default='./NDM_results')
parser.add_argument('--denoise_checkpoint', type=str, default='./NDM_checkpoint/NDM_LOL.pt')
parser.add_argument('--light_checkpoint', type=str, default='./light/outputs/checkpoints_light/LUM_100.pth')
opts = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
denoise_config = get_config(opts.denoise_config)

# Setup model and data loaderoots.trainer == 'UNIT':
DN_trainer = UNIT_Trainer(denoise_config).to(device)
state_dict = torch.load(opts.denoise_checkpoint, map_location='cpu')
DN_trainer.gen_x.load_state_dict(state_dict['x'])
DN_trainer.gen_y.load_state_dict(state_dict['y'])
DN_trainer.eval()
encode = DN_trainer.gen_x.encode_cont  # encode function
decode = DN_trainer.gen_y.decode_cont  # decode function

# pre-trained model set
light_config = get_config(opts.light_config)
light = DecomNet(light_config).to(device)
state_dict = torch.load(opts.light_checkpoint, map_location='cpu')
light.load_state_dict(state_dict)
light.eval()

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def display_all_results(original, clahe_image, model_output, title="Comparison"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    images = [original, clahe_image, model_output]
    titles = ["Original", "CLAHE", "Model Output"]
    for i in range(3):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')
    for i in range(3):
        axes[1, i].hist(images[i].ravel(), bins=256, range=[0, 256], color='gray')
        axes[1, i].set_title(f"Histogram ({titles[i]})")
        axes[1, i].set_xlim([0, 256])
    plt.tight_layout()
    plt.show()
imglist = os.listdir(opts.input_folder)
transform = transforms.Compose([transforms.ToTensor()])

for i, file in enumerate(imglist):
    print(f"Processing {file}...")
    filepath = os.path.join(opts.input_folder, file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = transform(Image.open(filepath).convert('RGB')).unsqueeze(0).to(device)

    h, w = image.size(2), image.size(3)
    pad_h, pad_w = h % 4, w % 4
    image = image[:, :, :h - pad_h, :w - pad_w]
    r_low, _ = light(image)
    content = encode(r_low)
    outputs = decode(content)
    original_image = (image.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    model_output = (r_low.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    if len(model_output.shape) == 3:
        model_output = cv2.cvtColor(model_output.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

    clahe_image = apply_clahe(original_gray)

    display_all_results(original_gray, clahe_image, model_output, title=file)

    name = os.path.splitext(file)[0]
    path = os.path.join(opts.output_folder, name + '.png')
    vutils.save_image(r_low.data, path)

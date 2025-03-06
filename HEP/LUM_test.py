from __future__ import print_function

import warnings

from utils import get_config
from trainer import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from models.LUM_model import DecomNet
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Light args setting')
parser.add_argument('--LUM_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, default='./test_images', help="input image path")
parser.add_argument('--output_folder', type=str, default='./LUM_results', help="output image path")
parser.add_argument('--LUM_checkpoint', type=str, default='./light/outputs/checkpoints_light/LUM_100.pth', help="checkpoint of light")
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

light_config = get_config(opts.LUM_config)
light = DecomNet(light_config)

state_dict = torch.load(opts.LUM_checkpoint, map_location='cpu', weights_only=True)

#state_dict = torch.load(opts.LUM_checkpoint, map_location='cpu')
light.load_state_dict(state_dict)
light.cuda()
light.eval()

if not os.path.exists(opts.input_folder):
    raise Exception('input path is not exists!')
imglist = os.listdir(opts.input_folder)
transform = transforms.Compose([transforms.ToTensor()])

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
def display_all_results(original, clahe_image, model_output, title="Comparison"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = [original, clahe_image, model_output]
    titles = ["Original", "CLAHE", "Model Output"]

    # Верхний ряд - изображения
    for i in range(3):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')

    # Нижний ряд - гистограммы
    for i in range(3):
        axes[1, i].hist(images[i].ravel(), bins=256, range=[0, 256], color='gray')
        axes[1, i].set_title(f"Histogram ({titles[i]})")
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()
for i, file in enumerate(imglist):
    print(file)
    filepath = opts.input_folder + '/' + file
    image = transform(Image.open(
        filepath).convert('RGB')).unsqueeze(0).cuda()
    # Start testing
    h, w = image.size(2), image.size(3)
    pad_h = h % 4
    pad_w = w % 4
    image = image[:, :, 0:h - pad_h, 0:w - pad_w]
    r_low, i_low = light(image)

    original_image = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    original_image = (original_image * 255).astype(np.uint8)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    # Применяем CLAHE
    clahe_image = apply_clahe(original_gray)

    # Преобразуем выход модели в чёрно-белое
    model_output = r_low.squeeze().cpu().detach().numpy()  # Удаляем размерность батча и используем .detach()
    if len(model_output.shape) == 3:
        model_output = cv2.cvtColor((model_output.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        model_output = (model_output * 255).astype(np.uint8)
    display_all_results(original_gray, clahe_image, model_output, title=file)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    outputs_back = r_low.clone()
    name = os.path.splitext(file)[0]
    path = os.path.join(opts.output_folder, name + '.png')
    vutils.save_image(r_low.data, path)

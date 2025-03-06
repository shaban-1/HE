from __future__ import print_function
import warnings
import os
import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from utils import get_config
from models.LUM_model import DecomNet

parser = argparse.ArgumentParser(description='Light args setting')
parser.add_argument('--LUM_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, default='./test_images', help="input image path")
parser.add_argument('--output_folder', type=str, default='./LUM_results', help="output image path")
parser.add_argument('--LUM_checkpoint', type=str, default='./light/outputs/checkpoints_light/LUM_100.pth',
                    help="checkpoint of light")
opts = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

os.makedirs(opts.output_folder, exist_ok=True)

light_config = get_config(opts.LUM_config)
light = DecomNet(light_config)

state_dict = torch.load(opts.LUM_checkpoint, map_location=device)
light.load_state_dict(state_dict)

light.to(device)
light.eval()

if not os.path.exists(opts.input_folder):
    raise Exception('Input path does not exist!')

imglist = os.listdir(opts.input_folder)
transform = transforms.Compose([transforms.ToTensor()])


def apply_clahe(image):
    """ Применяет CLAHE для улучшения контраста. """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def display_all_results(original, clahe_image, model_output, title="Comparison"):
    """ Отображает оригинальное изображение, результат CLAHE и выход модели с гистограммами. """
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
    print(f"Processing: {file}")

    filepath = os.path.join(opts.input_folder, file)

    image = transform(Image.open(filepath).convert('RGB')).unsqueeze(0).to(device)

    h, w = image.shape[2], image.shape[3]
    pad_h, pad_w = h % 4, w % 4
    image = image[:, :, :h - pad_h, :w - pad_w]

    with torch.no_grad():
        r_low, i_low = light(image)

    original_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    original_image = (original_image * 255).astype(np.uint8)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    clahe_image = apply_clahe(original_gray)

    model_output = r_low.squeeze(0).cpu().detach().numpy()
    if len(model_output.shape) == 3:
        model_output = cv2.cvtColor((model_output.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        model_output = (model_output * 255).astype(np.uint8)

    display_all_results(original_gray, clahe_image, model_output, title=file)

    name = os.path.splitext(file)[0]
    path = os.path.join(opts.output_folder, name + '.png')
    vutils.save_image(r_low.data, path)
    print(f"Saved result: {path}")

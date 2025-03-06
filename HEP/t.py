import os
from PIL import Image

input_folder = r"C:\Users\sevda\Job\LOL_dataset\train_gt"
output_folder = r"C:\Users\sevda\Job\LOL_dataset\train_gt_resized"

os.makedirs(output_folder, exist_ok=True)
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
target_size = (600, 400)

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size, Image.LANCZOS)  # Используем LANCZOS вместо ANTIALIAS
        img_resized.save(output_path)
    print(f"{image_file} resized to {target_size}")

print("All images resized successfully.")


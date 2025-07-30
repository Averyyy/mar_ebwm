import os
from PIL import Image

root_dir = '/work/hdd/bdta/aqian1/data/val-64-c7/'
target_size = (64, 64)

print(f"Searching for images in {root_dir}...")

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('.jpeg'):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    # Only resize if it's not already the target size
                    if img.size != target_size:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(file_path)
                        print(f"Resized: {file_path}")
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

print("Done.")
#!/usr/bin/env python3
import os
from PIL import Image
from torch_fidelity import calculate_metrics

def resize_images(src_dir, dst_dir, size=(64, 64)):
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        out_root = os.path.join(dst_dir, rel_path)
        os.makedirs(out_root, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(out_root, os.path.splitext(fname)[0] + '.png')
                with Image.open(src_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(size, Image.LANCZOS)
                    img.save(dst_path, format='PNG')

if __name__ == '__main__':
    src_root = '/work/nvme/belh/aqian1/imagenet-1k'
    resized_root = '/work/hdd/bdta/aqian1/mar_ebwm/data/imagenet-1k-64'
    stats_file = 'imagenet64_stats.npz'

    print(f"▶️ 64 resize\n   {resized_root}")
    resize_images(src_root, resized_root, size=(64, 64))

    print("▶️ fid stats…")
    metrics = calculate_metrics(
        input1=None,
        input2=resized_root,
        fid=True,
        kid=False,
        isc=False,
        prc=False,
        cuda=True,
        verbose=True,
        fid_statistics_file=stats_file,
    )

    print(f"✅ FID stats saved: {stats_file}")
    print("metrics:", metrics)

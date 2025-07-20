import glob
import os
from typing import Tuple

import torch
from torch.utils.data import Dataset


class CachedPTFolder(Dataset):
    """Dataset for latents cached as uncompressed .pt files.

    Each ``.pt`` file is a dictionary with keys ``'moments'`` and
    ``'moments_flip'`` (produced by ``engine_mar.cache_latents`` when
    ``--cache_format pt`` is selected).  The directory structure follows
    the original ImageNet layout so the class label can be inferred from
    the parent folder name, mirroring the behaviour of
    ``torchvision.datasets.ImageFolder``.
    """

    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt cache files found in {root}")

        # Map class folder names to integer labels (as ImageFolder does)
        class_names = sorted({os.path.basename(os.path.dirname(f)) for f in self.files})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.files[index]
        data = torch.load(path, map_location="cpu")
        # Random horizontal-flip counterpart to match original npz dataset
        if torch.rand(1) < 0.5:
            moments = data["moments"]
        else:
            moments = data["moments_flip"]

        # Determine label from parent directory name
        class_folder = os.path.basename(os.path.dirname(path))
        target = self.class_to_idx[class_folder]
        return moments, target 
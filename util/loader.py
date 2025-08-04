import os
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.datasets as datasets

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        
        # Handle corrupted images by skipping them
        try:
            sample = self.loader(path)
            # Force convert to RGB to handle various formats
            if sample.mode != 'RGB':
                sample = sample.convert('RGB')
        except (ValueError, OSError, IOError, Image.DecompressionBombError) as e:
            print(f"[Warning] Corrupted image {path}, error: {e}. Skipping and trying next image.")
            
            # Try next image in dataset
            index = (index + 1) % len(self.samples)
            return self.__getitem__(index)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        # data = np.load(path)
        try:
            data = np.load(path)
        except (EOFError, ValueError, OSError) as e:
            print(f"[Warning] Cant load {path}, error: {e}. Skipping")
            try:
                os.remove(path)
            except Exception:
                pass
            new_idx = (index + 1) % len(self)
            return self.__getitem__(new_idx)
        
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target

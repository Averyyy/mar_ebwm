import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Synthetic dataset that returns random images and labels.

    This is useful for profiling and debugging dataloader / GPU utilization
    bottlenecks. Images are randomly generated tensors in the range [-1, 1]
    with shape (3, img_size, img_size). Labels are random integers in
    [0, num_classes).

    The interface mimics ``torchvision.datasets.ImageFolder`` so it can be
    swapped in with minimal code changes.  When ``return_paths`` is True an
    additional dummy filename string is returned as the last element of the
    tuple, matching ``ImageFolderWithFilename`` used by the caching script.
    """

    def __init__(
        self,
        num_samples: int = 1281167,
        img_size: int = 256,
        num_classes: int = 1000,
        return_paths: bool = False,
    ):
        """Create a new synthetic dataset.

        Args:
            num_samples: Number of samples to generate (default: size of ImageNet train split).
            img_size: Height / width of the square synthetic image.
            num_classes: Number of classes – labels are sampled uniformly at random.
            return_paths: Whether to also return a dummy string representing the
                image filename (useful for caching code that expects it).
        """
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.return_paths = return_paths

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        # Generate a random image in [-1, 1]. Using rand() instead of randn()
        # avoids extremely large values that could cause numerical issues in
        # subsequent normalisation layers.
        img = torch.rand(3, self.img_size, self.img_size) * 2.0 - 1.0
        label = int(torch.randint(low=0, high=self.num_classes, size=(1,)).item())

        if self.return_paths:
            return img, label, f"synthetic/{index}.png"
        else:
            return img, label 
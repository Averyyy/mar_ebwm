import glob
import os
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedPTShard(Dataset):
    """Dataset for latents stored in large sharded *.pt files.

    Each shard is a dictionary with keys::
        {
            'moments':       Tensor [N, C, H, W]
            'moments_flip':  Tensor [N, C, H, W]
            'labels':        Tensor [N]
        }
    where *N* is the number of samples inside that shard (e.g. 20k).

    This format trades a small increase in file size for drastically fewer
    file-open calls, eliminating the I/O bubble that causes GPU utilisation
    spikes when training light-weight models such as DEBT.
    """

    def __init__(self, root: str, max_shards_in_ram: int = 4):
        super().__init__()
        self.root = root
        self.shard_paths: List[str] = sorted(glob.glob(os.path.join(root, "shard_*.pt")))
        if len(self.shard_paths) == 0:
            raise RuntimeError(f"No shard_*.pt files found under {root}. Have you run main_cache.py --cache_format ptshard?")

        # Build prefix sum of shard lengths without loading them fully
        self._shard_sizes: List[int] = []
        for p in self.shard_paths:
            hdr = torch.load(p, map_location="cpu", mmap=True)
            self._shard_sizes.append(hdr['labels'].numel())
            # Keep only meta to free RAM
            del hdr
        self._cum_sizes = np.cumsum([0] + self._shard_sizes)  # len = n_shards+1

        # LRU cache for loaded shards
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._lru: List[int] = []
        self._max_shards_in_ram = max_shards_in_ram
        
        # Preload first few shards to avoid cold start
        self._preload_shards()

    def __len__(self):
        return self._cum_sizes[-1]

    def _preload_shards(self):
        """Preload first few shards to avoid cold start delay."""
        num_preload = min(self._max_shards_in_ram, len(self.shard_paths))
        print(f"Preloading {num_preload} shards to warm up cache...")
        for i in range(num_preload):
            self._get_shard(i)
        print(f"Cache warmed up with {len(self._cache)} shards")

    def _get_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Return shard dict, loading it into RAM if necessary (simple LRU)."""
        if shard_idx in self._cache:
            # update LRU order
            self._lru.remove(shard_idx)
            self._lru.append(shard_idx)
            return self._cache[shard_idx]

        # Load shard
        shard = torch.load(self.shard_paths[shard_idx], map_location="cpu")
        self._cache[shard_idx] = shard
        self._lru.append(shard_idx)

        # Evict if cache too big
        if len(self._lru) > self._max_shards_in_ram:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return shard

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Locate shard via binary search over cumulative sizes
        shard_idx = int(np.searchsorted(self._cum_sizes, idx, side='right') - 1)
        local_idx = idx - self._cum_sizes[shard_idx]

        shard = self._get_shard(shard_idx)
        # Random hflip choice to match original behaviour
        if torch.rand(1) < 0.5:
            x = shard['moments'][local_idx]
        else:
            x = shard['moments_flip'][local_idx]
        y = int(shard['labels'][local_idx])
        return x, y 
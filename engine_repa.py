# engine_repa.py
# REPA utilities for representation evaluation of diffusion models (on-the-fly version)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from torch.utils.data import Dataset
from glob import glob

# -----------------------
# Cache Loader
# -----------------------

import os, glob, torch
from torch.utils.data import Dataset

class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir, split, layer_names):
        files = sorted(glob.glob(f"{cache_dir}/{split}/layer_rank*_shard*.pt"))

        self.layer_names = layer_names
        self.samples = {ln: [] for ln in layer_names}
        self.labels = []

        # load all shards
        for f in files:
            data = torch.load(f, map_location="cpu")
            for ln in layer_names:
                self.samples[ln].append(data[ln])
            self.labels.append(data["labels"])

        # concatenate everything
        self.samples = {ln: torch.cat(self.samples[ln], dim=0) for ln in layer_names}
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        """
        Returns:
            dict {layer_name: feature_tensor}, label
        """
        features = {ln: self.samples[ln][idx] for ln in self.layer_names}
        label = self.labels[idx]
        return features, label

# -----------------------
# Train probe using cached latents for a single layer
# -----------------------
def train_probe_cached(train_loader, layer_name, num_classes,
                             epochs=10, lr=1e-3, device="cuda"):
    # Get feature dimension from first batch
    xb, _ = next(iter(train_loader))
    feat_dim = xb[layer_name].shape[1]

    probe = LinearProbe(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    def lr_lambda(ep):
        return 0.5 * (1 + math.cos(math.pi * ep / epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for ep in range(epochs):
        probe.train()
        for feats, labels in tqdm(train_loader, desc=f"train ep{ep} [cached]"):
            xb_layer = feats[layer_name].to(device)
            yb = labels.to(device)
            logits = probe(xb_layer)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return probe

# -----------------------
# Evaluate probe using cached latents for a single layer
# -----------------------
@torch.no_grad()
def eval_probe_cached(val_loader, probe, layer_name, device="cuda"):
    probe.eval()
    all_preds, all_labels = [], []
    for feats, labels in tqdm(val_loader, desc=f"eval [cached]"):
        xb_layer = feats[layer_name].to(device)
        yb = labels.to(device)
        logits = probe(xb_layer)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return accuracy_score(all_labels, all_preds)




# -----------------------
# Feature grabber with forward hooks
# -----------------------
class FeatureGrabber:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.handles = []
        self.features = {name: [] for name in layer_names}

    def _get_module(self, name):
        cur = self.model
        for part in name.split('.'):
            cur = getattr(cur, part)
        return cur

    def register(self):
        def make_hook(name):
            def hook(module, inp, out):
                self.features[name].append(out.detach())
            return hook

        for name in self.layer_names:
            mod = self._get_module(name)
            h = mod.register_forward_hook(make_hook(name))
            self.handles.append(h)

    def clear(self):
        for k in self.features.keys():
            self.features[k] = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self):
        return {k: torch.cat(v, dim=0) if len(v) > 0 else None
                for k, v in self.features.items()}


# -----------------------
# Pooling utility
# -----------------------
def spatial_pool(features, mode='global_mean'):
    if features is None:
        return None
    if features.ndim == 2:
        return features
    if mode == 'global_mean':
        return features.mean(axis=1)
    elif mode == 'global_max':
        return features.max(axis=1)
    else:
        N, P, D = features.shape
        return features.reshape(N, P * D)


# -----------------------
# Linear probe
# -----------------------
class LinearProbe(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(self.bn(x))


# -----------------------
# Train probe on-the-fly
# -----------------------
def train_probe_onthefly(diff_model, vae, dataloader, layer, num_classes,
                         epochs=10, lr=1e-3, device="cuda", pool_mode="global_mean"):
    device = torch.device(device)
    grabber = FeatureGrabber(diff_model, [layer])
    grabber.register()

    diff_model.eval()
    vae.eval()

    # infer feature dimension from one batch
    with torch.no_grad():
        imgs, labels = next(iter(dataloader))
        imgs, labels = imgs.to(device), labels.to(device)
        latents = vae.encode(imgs).sample().mul_(0.2325)
        _ = diff_model(latents, torch.zeros(imgs.size(0), dtype=torch.long, device=device), y=labels)
        feats = grabber.get()[layer].cpu()
        pooled = spatial_pool(feats.numpy(), mode=pool_mode)
        feat_dim = pooled.shape[1]
    grabber.clear()

    # probe
    probe = LinearProbe(feat_dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    def lr_lambda(ep):
        return 0.5 * (1 + math.cos(math.pi * ep / epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # training loop
    for ep in range(epochs):
        probe.train()
        for imgs, labels in tqdm(dataloader, desc=f"train ep{ep} [{layer}]"):
            imgs, labels = imgs.to(device), labels.to(device)
            latents = vae.encode(imgs).sample().mul_(0.2325)

            _ = diff_model(latents, torch.zeros(imgs.size(0), dtype=torch.long, device=device), y=labels)
            feats = grabber.get()[layer]
            grabber.clear()

            pooled = spatial_pool(feats.cpu().numpy(), mode=pool_mode)
            xb = torch.tensor(pooled, dtype=torch.float32, device=device)
            yb = labels

            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    grabber.remove()
    return probe


# -----------------------
# Evaluate probe on-the-fly
# -----------------------
@torch.no_grad()
def eval_probe_onthefly(diff_model, vae, dataloader, layer, probe,
                        device="cuda", pool_mode="global_mean"):
    grabber = FeatureGrabber(diff_model, [layer])
    grabber.register()
    probe.eval()

    all_preds, all_labels = [], []
    for imgs, labels in tqdm(dataloader, desc=f"eval [{layer}]"):
        imgs, labels = imgs.to(device), labels.to(device)
        latents = vae.encode(imgs).sample().mul_(0.2325)

        _ = diff_model(latents, torch.zeros(imgs.size(0), dtype=torch.long, device=device), y=labels)
        feats = grabber.get()[layer]
        grabber.clear()

        pooled = spatial_pool(feats.cpu().numpy(), mode=pool_mode)
        xb = torch.tensor(pooled, dtype=torch.float32, device=device)
        logits = probe(xb)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    grabber.remove()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return accuracy_score(all_labels, all_preds)


# -----------------------
# Main evaluation
# -----------------------
def evaluate_model_representation(diff_model, vae, train_loader, val_loader,
                                  layer_names, device="cuda",
                                  num_classes=1000, linear_epochs=10,
                                  pool_mode="global_mean"):
    results = {'linear_probe': {}}
    for layer in layer_names:
        probe = train_probe_onthefly(diff_model, vae, train_loader,
                                     layer, num_classes,
                                     epochs=linear_epochs,
                                     device=device,
                                     pool_mode=pool_mode)
        acc = eval_probe_onthefly(diff_model, vae, val_loader,
                                  layer, probe,
                                  device=device,
                                  pool_mode=pool_mode)
        results['linear_probe'][layer] = acc
    return results


def cache_layers(diff_model, vae, data_loader, layer_names, device, args=None, pool_mode="global_mean", typ="train"):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.makedirs(os.path.join(args.cached_path, typ), exist_ok=True)

    grabber = FeatureGrabber(diff_model, layer_names)
    grabber.register()
    diff_model.eval()
    vae.eval()

    shard_features = {ln: [] for ln in layer_names}
    shard_labels = []
    shard_id = 0

    for step, (imgs, labels) in enumerate(tqdm(data_loader, desc=f"Rank {rank} caching")):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            latents = vae.encode(imgs).sample().mul_(0.2325)
            _ = diff_model(latents, torch.zeros(imgs.size(0), dtype=torch.long, device=device), y=labels)
            feats = grabber.get()
            grabber.clear()

        for ln in layer_names:
            pooled = spatial_pool(feats[ln].cpu(), mode=pool_mode)
            shard_features[ln].append(pooled)
        shard_labels.append(labels.cpu())

        if (step + 1) % getattr(args, "cache_shard_size", 2000) == 0:
            save_dict = {ln: torch.cat(shard_features[ln]) for ln in layer_names}
            save_dict["labels"] = torch.cat(shard_labels)
            save_path = os.path.join(args.cached_path, f'{typ}', f"layer_rank{rank:02d}_shard{shard_id:05d}.pt")
            torch.save(save_dict, save_path)
            shard_id += 1
            shard_features = {ln: [] for ln in layer_names}
            shard_labels = []
        dist.barrier() 
    if shard_labels:
        save_dict = {ln: torch.cat(shard_features[ln]) for ln in layer_names}
        save_dict["labels"] = torch.cat(shard_labels)
        save_path = os.path.join(args.cached_path, f'{typ}', f"layer_rank{rank:02d}_shard{shard_id:05d}.pt")
        torch.save(save_dict, save_path)

    grabber.remove()
    dist.barrier() 

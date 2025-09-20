# engine_repa.py
# REPA utilities for representation evaluation of diffusion models
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cdist


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
                self.features[name].append(out.detach().cpu())
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

    def concat(self):
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
    elif mode == 'global_max':  # keep for optional use
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


def train_linear_probe(train_loader, val_loader, dim, num_classes,
                       epochs=90, lr=1e-3, device='cuda'):
    device = torch.device(device)
    model = LinearProbe(dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(ep):
        return 0.5 * (1 + math.cos(math.pi * ep / epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for ep in tqdm(range(epochs), desc="epochs"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.long)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    # eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, dtype=torch.float32)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc


# -----------------------
# CKNNA utilities
# -----------------------
def compute_knn_indices(features, k=10, metric='cosine'):
    dist = cdist(features, features, metric=metric)
    np.fill_diagonal(dist, np.inf)
    knn_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
    return knn_idx


def row_centered_inner(features):
    G = features @ features.T
    row_means = G.mean(axis=1, keepdims=True)
    return G - row_means


def compute_align(A, B, k=10):
    N = A.shape[0]
    knn_A = compute_knn_indices(A, k=k)
    knn_B = compute_knn_indices(B, k=k)
    M_A = row_centered_inner(A)
    M_B = row_centered_inner(B)
    total = 0.0
    for i in tqdm(range(N), desc='CKNN'):
        inter = set(knn_A[i]).intersection(set(knn_B[i]))
        if len(inter) == 0:
            continue
        js = np.fromiter(inter, dtype=int)
        total += (M_A[i, js] * M_B[i, js]).sum()
    return total / ((N - 1) ** 2)


def compute_cknna(A, B, k=10):
    ab = compute_align(A, B, k)
    aa = compute_align(A, A, k)
    bb = compute_align(B, B, k)
    return ab / math.sqrt(max(aa * bb, 1e-12))


# -----------------------
# Feature extraction
# -----------------------
@torch.no_grad()
def extract_features(diff_model, vae, dataloader, layer_names,
                     device='cuda', timestep=0):
    diff_model.eval()
    vae.eval()
    grabber = FeatureGrabber(diff_model, layer_names)
    grabber.register()
    grabber.clear()

    device = torch.device(device)
    for imgs, labels in tqdm(dataloader, desc="extract diffusion feats"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        posterior = vae.encode(imgs)
        latents = posterior.sample().mul_(0.2325)
        _ = diff_model(
            latents,
            torch.tensor([timestep] * imgs.shape[0], device=device, dtype=torch.long),
            y=labels
        )

    feats = grabber.concat()
    grabber.remove()
    return {k: v.cpu().numpy() if v is not None else None for k, v in feats.items()}


# -----------------------
# Main evaluation
# -----------------------
def evaluate_model_representation(diff_model, pretrained_encoder,
                                  train_dataloader, val_dataloader, layer_names,
                                  vae=None,
                                  device='cuda',
                                  cknn=False,
                                  cknn_k=10, pool_mode='global_mean',
                                  linear_epochs=90, linear_bs=128):
    device = torch.device(device)
    results = {'cknna': {}, 'linear_probe': {}}

    # -----------------------
    # 1. CKNNA: encoder vs diffusion features
    # -----------------------
    if cknn:
        # encoder (target) features
        all_feats, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(train_dataloader, desc="encoder feats (train)"):
                imgs = imgs.to(device)
                out = pretrained_encoder(imgs)
                if hasattr(out, "last_hidden_state"):
                    feat = out.last_hidden_state
                elif hasattr(out, "pooler_output"):
                    feat = out.pooler_output.unsqueeze(1)
                else:
                    feat = out
                all_feats.append(feat.detach().cpu().numpy())
                all_labels.append(labels.numpy())
        target_feats = np.concatenate(all_feats, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        pooled_target = spatial_pool(target_feats, mode=pool_mode)

    # -----------------------
    # 2. Diffusion features (train + val)
    # -----------------------
    diff_train_feats = extract_features(diff_model, vae, train_dataloader, layer_names, device=device)
    diff_val_feats   = extract_features(diff_model, vae, val_dataloader, layer_names, device=device)

    # collect labels for probe
    train_labels, val_labels = [], []
    for _, y in train_dataloader:
        train_labels.append(y.numpy())
    for _, y in val_dataloader:
        val_labels.append(y.numpy())
    train_labels = np.concatenate(train_labels)
    val_labels   = np.concatenate(val_labels)
    num_classes = int(max(train_labels.max(), val_labels.max())) + 1

    # -----------------------
    # 3. Per-layer evaluation
    # -----------------------
    for layer in tqdm(layer_names, desc="Layers"):
        feats_train = diff_train_feats[layer]
        feats_val   = diff_val_feats[layer]
        if feats_train is None or feats_val is None:
            results['cknna'][layer] = None
            results['linear_probe'][layer] = None
            continue

        pooled_train = spatial_pool(feats_train, mode=pool_mode)
        pooled_val   = spatial_pool(feats_val, mode=pool_mode)
        feat_dim = pooled_train.shape[1]

        # build datasets for probe
        train_ds = TensorDataset(torch.tensor(pooled_train, dtype=torch.float32),
                                 torch.tensor(train_labels, dtype=torch.long))
        val_ds   = TensorDataset(torch.tensor(pooled_val, dtype=torch.float32),
                                 torch.tensor(val_labels, dtype=torch.long))
        train_loader_probe = DataLoader(train_ds, batch_size=linear_bs, shuffle=True)
        val_loader_probe   = DataLoader(val_ds, batch_size=linear_bs, shuffle=False)

        # train + eval linear probe
        acc = train_linear_probe(train_loader_probe, val_loader_probe,
                                 dim=feat_dim, num_classes=num_classes,
                                 epochs=linear_epochs, device=device)
        results['linear_probe'][layer] = acc

        # cknna
        if cknn:
            ckn = compute_cknna(pooled_train, pooled_target, k=cknn_k)
            results['cknna'][layer] = ckn
        else:
            results['cknna'][layer] = None

    return results

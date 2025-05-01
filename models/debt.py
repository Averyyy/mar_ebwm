import torch
import torch.nn as nn
import numpy as np
from functools import partial

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs


class DEBT(nn.Module):
    """Diffusion Energy‑Based Transformer (DEBT) – incremental fixes.
    """

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        embed_dim=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        class_num=1000,
        dropout_prob=0.1,
        mcmc_num_steps=2,
        mcmc_step_size=0.01,
        langevin_dynamics_noise=0.0,
        denoising_initial_condition="random_noise",
        double_condition=False,
        training=True,
    ):
        super().__init__()

        # -------- image / patch ----------
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = 16
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w  # S
        self.token_embed_dim = 16 * patch_size ** 2
        self.embed_dim = embed_dim

        self.mcmc_num_steps = mcmc_num_steps
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.denoising_initial_condition = denoising_initial_condition
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=True)

        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.seq_len + 2, embed_dim))
        self.start_token = nn.Parameter(torch.randn(1, embed_dim))
        self.end_token = nn.Parameter(torch.randn(1, embed_dim))
        self.double_condition = double_condition

        self.input_proj = nn.Linear(self.token_embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, self.token_embed_dim)

        t_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            ffn_dim_multiplier=mlp_ratio,
            adaln_zero_init=False,
            max_seq_len=2 * self.seq_len + 2,
            final_output_dim=1,
        )
        self.transformer = EBTAdaLN(params=t_args, max_mcmc_steps=mcmc_num_steps)
        self.training = training

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.y_embedder.weight, 0, 0.02)
        nn.init.normal_(self.pos_embed, 0, 0.02)
        nn.init.normal_(self.start_token, 0, 0.02)
        nn.init.normal_(self.end_token, 0, 0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        return x.reshape(B, h * w, -1)

    def unpatchify(self, x):
        B = x.size(0)
        p = self.patch_size
        c = self.vae_embed_dim
        h = w = self.seq_h
        x = x.reshape(B, h, w, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        return x.reshape(B, c, h * p, w * p)

    def corrupt_embeddings(self, ref_like):
        if isinstance(ref_like, torch.Tensor):
            shape = list(ref_like.shape)
            device, dtype = ref_like.device, ref_like.dtype
        else:
            shape, device, dtype = list(ref_like), None, torch.float32
        shape[-1] = self.embed_dim
        if self.denoising_initial_condition == "zeros":
            return torch.zeros(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)

    def refine_embeddings(self, real_embeddings, predicted_embeddings, c):
        if predicted_embeddings.shape[-1] == self.token_embed_dim:
            predicted_embeddings = self.input_proj(predicted_embeddings)

        alpha = torch.clamp(self.alpha, min=1e-4)
        B = real_embeddings.size(0)
        s_tok = self.start_token.to(real_embeddings.dtype).unsqueeze(1).expand(B, 1, -1)
        e_tok = self.end_token.to(real_embeddings.dtype).unsqueeze(1).expand(B, 1, -1)
        with torch.enable_grad():
            for step in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                all_emb = torch.cat([s_tok, real_embeddings, predicted_embeddings, e_tok], 1)
                pos_slice = self.pos_embed[:, : all_emb.size(1), :]
                all_emb = all_emb + pos_slice
                energy = self.transformer(all_emb, 0, mcmc_step=step, c=c)
                energy_sum = energy.sum()
                if self.training:
                    grad = torch.autograd.grad(energy_sum, predicted_embeddings, create_graph=True, retain_graph=True)[0]
                else:
                    grad = torch.autograd.grad(energy_sum, predicted_embeddings)[0]
                predicted_embeddings = predicted_embeddings - alpha * grad
                # if (not self.training) and self.langevin_dynamics_noise > 0:
                #     predicted_embeddings += torch.randn_like(predicted_embeddings) * self.langevin_dynamics_noise
        return predicted_embeddings.detach()

    def forward(self, x_start, labels):
        gt_tokens = self.patchify(x_start)
        real_emb = self.input_proj(gt_tokens)
        pred_emb = self.corrupt_embeddings(real_emb)
        c = self.y_embedder(labels)
        pred_emb = self.refine_embeddings(real_emb, pred_emb, c)
        pred_tokens = self.output_proj(pred_emb)
        recon = self.unpatchify(pred_tokens)
        return ((recon - x_start) ** 2).mean()

    @torch.no_grad()
    def sample_tokens(self, bsz, num_iter=None, cfg=1.0, labels=None, temperature=1.0, progress=False):
        print("sampling started")
        if num_iter is None:
            num_iter = self.seq_len
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)
        c = self.y_embedder(labels)

        prefix_embeds = torch.zeros(bsz, 0, self.embed_dim, device=device)
        for iter in range(num_iter):
            if iter % 2 == 0:
                print(f"iter {iter} / {num_iter}")
            noise_tok = self.corrupt_embeddings(torch.zeros(bsz, 1, self.embed_dim, device=device))
            pred_seq = torch.cat([prefix_embeds, noise_tok], 1) 

            dummy_zero = torch.zeros_like(noise_tok)
            real_seq = torch.cat([prefix_embeds, dummy_zero], 1)

            refined = self.refine_embeddings(real_seq, pred_seq, c)
            new_token = refined[:, -1:]
            prefix_embeds = torch.cat([prefix_embeds, new_token], 1)

        pred_tokens = self.output_proj(prefix_embeds)                    # (B,S,token_dim)
        return self.unpatchify(pred_tokens)


if __name__ == "__main__":
    model = DEBT()
    x = torch.randn(2, 16, 16, 16)
    labels = torch.tensor([0, 1])
    # print("loss:", model(x, labels).item())
    model.training = False
    print("sampled:", model.sample_tokens(2, labels=labels).shape)

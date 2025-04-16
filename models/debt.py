import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs

class DEBT(nn.Module):
    """
    Diffusion Energy-Based Transformer (DEBT) 
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
        mcmc_num_steps=10,
        mcmc_step_size=0.1,
        langevin_dynamics_noise=0.0,
        denoising_initial_condition='random_noise',
        double_condition=False,
    ):
        super().__init__()

        # Image and patch dimensions
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = 16
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = 16 * patch_size**2  # VAE latent dimension * patch_size^2
        self.embed_dim = embed_dim
        self.mcmc_num_steps = mcmc_num_steps
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.denoising_initial_condition = denoising_initial_condition

        # MCMC step size learnable
        self.alpha = nn.Parameter(torch.tensor(mcmc_step_size), requires_grad=True)

        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.seq_len + 2, embed_dim))  # +2 用于 <s> 和 </s>

        self.start_token = nn.Parameter(torch.randn(1, embed_dim))
        self.end_token = nn.Parameter(torch.randn(1, embed_dim))

        self.input_proj = nn.Linear(self.token_embed_dim, embed_dim, bias=True)
        self.output_proj = nn.Linear(embed_dim, self.token_embed_dim, bias=True)

        self.dropout_prob = dropout_prob
        self.double_condition = double_condition

        # Create EBT transformer with AdaLN
        transformer_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            ffn_dim_multiplier=mlp_ratio,
            adaln_zero_init=False,
            max_seq_len=2 * self.seq_len + 2,  # <s> + x_gt + predicted_embeddings + </s>
            final_output_dim=1,  # Energy prediction
        )

        self.transformer = EBTAdaLN(
            params=transformer_args,
            max_mcmc_steps=mcmc_num_steps
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize embedders
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

        nn.init.normal_(self.start_token, std=0.02)
        nn.init.normal_(self.end_token, std=0.02)

    def patchify(self, x):
        """Convert a batch of latent embeddings to patches"""
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        """Convert patches back to a batch of latent embeddings"""
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def corrupt_embeddings(self, shape):
        """Corrupt embeddings to initialize predicted_embeddings."""
        if self.denoising_initial_condition == "random_noise":
            return torch.randn(shape, device=shape.device)
        elif self.denoising_initial_condition == "zeros":
            return torch.zeros(shape, device=shape.device)
        else:
            raise ValueError(f"{self.denoising_initial_condition} not supported")

    def refine_embeddings(self, real_embeddings, initial_predicted_embeddings, c, mcmc_step):
        predicted_embeddings = initial_predicted_embeddings
        alpha = torch.clamp(self.alpha, min=0.0001)
        for mcmc_step in range(self.mcmc_num_steps ):
            with torch.set_grad_enabled(True):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                # <s> + real_embeddings + predicted_embeddings + </s>
                if self.double_condition:
                    start_token = self.start_token.expand(real_embeddings.shape[0], -1) + c
                else:
                    start_token = self.start_token.expand(real_embeddings.shape[0], -1)
                end_token = self.end_token.expand(real_embeddings.shape[0], -1)
                all_embeddings = torch.cat([start_token, real_embeddings, predicted_embeddings, end_token], dim=1)
                all_embeddings = all_embeddings + self.pos_embed
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=c)
                energy_sum = energy_preds[:, self.seq_len + 1 : -1].sum() #exclude </s>
                grad = torch.autograd.grad(
                    energy_sum, predicted_embeddings, create_graph=True, retain_graph=True
                )[0]

                predicted_embeddings = predicted_embeddings - alpha * grad

                if not self.training and self.langevin_dynamics_noise > 0:
                    noise = torch.randn_like(predicted_embeddings) * self.langevin_dynamics_noise
                    predicted_embeddings = predicted_embeddings + noise
        return predicted_embeddings

    def forward(self, x_start, labels):
        """
        Args:
            x_start: (B, C, H, W) - Clean image
            labels: (B,) - Class labels
        """
        bsz = x_start.shape[0]
        device = x_start.device

        x_gt_patches = self.patchify(x_start)  # [B, S, token_embed_dim]
        real_embeddings = self.input_proj(x_gt_patches)  # [B, S, embed_dim]

        initial_predicted_embeddings = self.corrupt_embeddings(x_gt_patches.shape)

        y_emb = self.y_embedder(labels)
        c = y_emb

        predicted_embeddings = self.refine_embeddings(real_embeddings, initial_predicted_embeddings, c, mcmc_step=0)

        predicted_tokens = self.output_proj(predicted_embeddings)  # [B, S, token_embed_dim]
        x_0_pred = self.unpatchify(predicted_tokens)  # [B, C, H, W]

        loss = ((x_0_pred - x_start) ** 2).mean()
        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, labels=None, temperature=1.0, progress=False):
        """
        Autoregressively sample tokens from the model.
        Returns:
            x_0_pred: generated image in latent space
        """
        device = next(self.parameters()).device
        shape = (bsz, self.seq_len, self.token_embed_dim)
        predicted_embeddings = torch.zeros(bsz, 0, self.token_embed_dim, device=device)

        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)

        y_emb = self.y_embedder(labels)
        c = y_emb

        for step in range(self.seq_len):
            new_token = self.corrupt_embeddings((bsz, 1, self.token_embed_dim))
            current_predicted = torch.cat([predicted_embeddings, new_token], dim=1)

            real_embeddings = torch.zeros(bsz, self.seq_len, self.embed_dim, device=device)

            # 优化整个序列
            current_predicted = self.refine_embeddings(real_embeddings, current_predicted, c, mcmc_step=step)

            # 将最后一个优化后的token追加到序列中
            predicted_embeddings = torch.cat([predicted_embeddings, current_predicted[:, -1:]], dim=1)

        # 还原为图像潜在表示
        x_0_pred = self.unpatchify(predicted_embeddings)
        return x_0_pred

if __name__ == "__main__":
    # 示例用法
    model = DEBT()
    x = torch.randn(2, 16, 16, 16)  # 模拟VAE潜在表示
    labels = torch.tensor([0, 1])
    loss = model(x, labels)
    print(f"训练损失: {loss.item()}")

    # 采样示例
    sampled_images = model.sample_tokens(bsz=2, labels=labels)
    print(f"采样图像形状: {sampled_images.shape}")
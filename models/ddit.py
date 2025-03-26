import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs
from diffusion import create_diffusion


class TimestepEmbedder(nn.Module):
    # borrowed from mar
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        # Shape: [B, D]
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Shape: [B, D]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class DDiT(nn.Module):
    """
    Decoder-only Diffusion Transformer (DDiT) model.
    """

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        embed_dim=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.,
        class_num=1000,
        dropout_prob=0.1,
        learn_sigma=False,
        num_sampling_steps='100',
        diffusion_batch_mul=4,
    ):
        super().__init__()

        # Image and patch dimensions
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = 16
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.full_seq_len = (self.seq_len + 1) * 2
        
        self.token_embed_dim = 16 * patch_size**2  # VAE latent dimension * patch size^2
        self.embed_dim = embed_dim
        self.out_channels = 16 * 2 if learn_sigma else 16
        self.diffusion_batch_mul = diffusion_batch_mul

        # Timestep and class embedders
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)
        
        # pos embed learnable
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 2, embed_dim))
        
        self.start_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.end_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.input_proj = nn.Linear(self.token_embed_dim, embed_dim, bias=True)
        self.output_proj = nn.Linear(embed_dim, self.out_channels * patch_size**2, bias=True)
        
        self.dropout_prob = dropout_prob

        # Create EBT transformer with AdaLN
        transformer_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            ffn_dim_multiplier=mlp_ratio,
            adaln_zero_init=False,
            max_seq_len=self.full_seq_len,
            final_output_dim=embed_dim,
        )

        self.transformer = EBTAdaLN(
            params=transformer_args,
            max_mcmc_steps=None  # Placeholder value for EBTAdaLN
        )

        # Diffusion process
        self.diffusion = create_diffusion(
            timestep_respacing=num_sampling_steps,
            noise_schedule="cosine",
            learn_sigma=learn_sigma,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize embedders
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        nn.init.normal_(self.start_embed, std=0.02)
        nn.init.normal_(self.end_embed, std=0.02)

        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def patchify(self, x):
        """Convert a batch of latent embeddings to patches"""
        # Input shape: [B, C, H, W], Output shape: [B, S, C*P*P]
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        """Convert patches back to a batch of latent embeddings"""
        # Input shape: [B, S, C*P*P], Output shape: [B, C, H, W]
        bsz = x.shape[0]
        p = self.patch_size
        c = self.out_channels  # VAE latent dimension
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def forward(self, x_start, labels):
        """        
        Args:
            x_start: (B, C, H, W)
            labels: (B,) 
        """
        bsz = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.diffusion.num_timesteps, (bsz,), device=device)
        
        loss_dict = self.diffusion.training_losses(
            model=self._forward_impl,
            x_start=x_start,
            t=t,
            model_kwargs={"x_start": x_start, "labels": labels,},
        )
        
        return loss_dict["loss"].mean()

    def _forward_impl(self, x, t, x_start, labels):
        """
        returns ebt output 
        """
        # Patchify the input x_t, gt - [B, S, D_patch]
        x_start_patches = self.patchify(x_start)
        x_t_patches = self.patchify(x)


        # Project to embedding dim - [B, S, embed_dim]
        x_start_embed = self.input_proj(x_start_patches)
        x_t_embed = self.input_proj(x_t_patches)
        
        real_tokens = torch.cat([
            self.start_embed.expand(x_start_embed.shape[0], -1, -1),
            x_start_embed
        ], dim=1)  # [B, S+1, D]
        
        pred_tokens = torch.cat([
            x_t_embed,
            self.end_embed.expand(x_t_embed.shape[0], -1, -1)
        ], dim=1)  # [B, S+1, D]
        
        real_tokens = real_tokens + self.pos_embed[:, :self.seq_len + 1, :]  # Positions 0 to seq_len
        pred_tokens = pred_tokens + self.pos_embed[:, 1:self.seq_len + 2, :] # Positions 1 to seq_len+1
        
        combined = torch.cat([real_tokens, pred_tokens], dim=1)

        # Get timestep and class embeddings - [B, embed_dim]
        t_emb = self.t_embedder(t)

        if self.training and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, torch.ones_like(labels) * self.y_embedder.num_embeddings - 1, labels)
        
        y_emb = self.y_embedder(labels)
        c = t_emb + y_emb  # [B, embed_dim]

        
        # Forward pass through ebt
        transformer_output = self.transformer(combined, start_pos=0, mcmc_step=None, c=c)

        # Project back to token space [B, S, C*P*P]
        token_preds = self.output_proj(transformer_output[:, :self.seq_len, :])  # [B, seq_len, out_channels * patch_size^2]
        # Reshape and unpatchify to get the final output [B, C, H, W]
        final_output = self.unpatchify(token_preds)

        return final_output
    def forward_with_cfg(self, x, t, labels, cfg_scale):
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        t_combined = torch.cat([t[:len(t) // 2], t[:len(t) // 2]], dim=0)

        # Conditional and unconditional labels
        labels_cond = labels[:len(labels) // 2]
        labels_uncond = torch.full_like(labels_cond, self.y_embedder.num_embeddings - 1)  # 无条件标签
        labels_combined = torch.cat([labels_cond, labels_uncond], dim=0)

        # Forward pass with current noise as x_start
        model_output = self._forward_impl(combined, t_combined, x_start=combined, labels=labels_combined)

        # Split conditional and unconditional outputs
        model_output_cond, model_output_uncond = torch.split(model_output, len(model_output) // 2, dim=0)

        # Apply CFG formula
        guided_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)

        return guided_output

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        device = next(self.parameters()).device
        shape = (bsz, self.vae_embed_dim, self.img_size // self.vae_stride, self.img_size // self.vae_stride)
        
        img = torch.randn(*shape, device=device)

        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)

        def model_fn(x, ts):
            if cfg > 1.0:
                return self.forward_with_cfg(x, ts, labels, cfg_scale=cfg)
            else:
                return self._forward_impl(x, ts, x_start=x, labels=labels)

        samples = self.diffusion.p_sample_loop(
            model_fn,
            shape,
            noise=img,
            device=device,
            progress=progress
        )

        return samples

import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs
from diffusion import create_diffusion

class TimestepEmbedder(nn.Module):
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DEBT(nn.Module):
    """
    Diffusion Energy-Based Transformer (DEBT) model.
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
        learn_sigma=False,
        num_sampling_steps='100',
        diffusion_batch_mul=4,
        mcmc_num_steps=10,
        mcmc_step_size=0.1,
        langevin_dynamics_noise=0.01,
        denoising_initial_condition='random_noise',
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
        self.out_channels = 16 * 2 if learn_sigma else 16
        self.diffusion_batch_mul = diffusion_batch_mul
        self.mcmc_num_steps = mcmc_num_steps
        self.mcmc_step_size = mcmc_step_size
        self.langevin_dynamics_noise = langevin_dynamics_noise
        self.denoising_initial_condition = denoising_initial_condition

        # Timestep and class embedders
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.seq_len, embed_dim))

        self.input_proj = nn.Linear(self.token_embed_dim, embed_dim, bias=True)
        self.output_proj = nn.Linear(embed_dim, self.token_embed_dim, bias=True)

        self.dropout_prob = dropout_prob

        # Create EBT transformer with AdaLN
        transformer_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            ffn_dim_multiplier=mlp_ratio,
            adaln_zero_init=False,
            max_seq_len=2 * self.seq_len,  # gt + predicted embeddings
            final_output_dim=1,  # Energy prediction
        )

        self.transformer = EBTAdaLN(
            params=transformer_args,
            max_mcmc_steps=mcmc_num_steps
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

        # Initialize projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

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

    def corrupt_embeddings(self, embeddings):
        """Corrupt embeddings to initialize predicted_embeddings."""
        if self.denoising_initial_condition == "most_recent_embedding":
            predicted_embeddings = embeddings.clone()
        elif self.denoising_initial_condition == "random_noise":
            predicted_embeddings = torch.randn_like(embeddings)
        elif self.denoising_initial_condition == "zeros":
            predicted_embeddings = torch.zeros_like(embeddings)
        else:
            raise ValueError(f"{self.denoising_initial_condition} not supported")
        return predicted_embeddings

    def refine_embeddings(self, real_embeddings, initial_predicted_embeddings, c, t):
        """Refine predicted embeddings using MCMC to minimize energy."""
        predicted_embeddings = initial_predicted_embeddings
        alpha = max(self.mcmc_step_size, 0.0001)
        for mcmc_step in range(self.mcmc_num_steps):
            with torch.set_grad_enabled(True):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                all_embeddings = torch.cat([real_embeddings, predicted_embeddings], dim=1)
                all_embeddings = all_embeddings + self.pos_embed
                energy_preds = self.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step, c=c)
                energy_sum = energy_preds[:, self.seq_len:].sum()
                predicted_embeds_grad = torch.autograd.grad(
                    energy_sum, predicted_embeddings, create_graph=True, retain_graph=True
                )[0]
                predicted_embeddings = predicted_embeddings - alpha * predicted_embeds_grad
                if not self.training and self.langevin_dynamics_noise > 0:
                    noise = torch.randn_like(predicted_embeddings) * self.langevin_dynamics_noise
                    predicted_embeddings = predicted_embeddings + noise

        return predicted_embeddings

    def _forward_impl(self, x_t, t, labels, x_start=None):
        """
        Forward pass for diffusion training/inference.
        Args:
            x_t: Noisy image at timestep t, shape [B, C, H, W]
            t: Timestep, shape [B]
            labels: Class labels, shape [B]
            x_start: Clean image (only during training), shape [B, C, H, W]
        Returns:
            Predicted clean image (x_0), shape [B, C, H, W]
        """
        # Patchify x_t and project to embeddings
        x_t_patches = self.patchify(x_t)
        x_t_embeddings = self.input_proj(x_t_patches)  # [B, S, embed_dim]

        # Real embeddings
        if self.training and x_start is not None:
            x_start_patches = self.patchify(x_start)
            real_embeddings = self.input_proj(x_start_patches)
        else:
            real_embeddings = torch.zeros_like(x_t_embeddings)  # No x_start during inference

        # Initialize predicted_embeddings
        if self.training and x_start is not None:
            # Corrupt real_embeddings for training
            initial_predicted_embeddings = self.corrupt_embeddings(real_embeddings)
        else:
            # Use random noise or zeros for inference, not x_t_embeddings directly
            initial_predicted_embeddings = self.corrupt_embeddings(x_t_embeddings * 0)  # Shape match

        # Timestep and class embeddings
        t_emb = self.t_embedder(t)
        if self.training and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, torch.ones_like(labels) * self.y_embedder.num_embeddings - 1, labels)
        y_emb = self.y_embedder(labels)
        c = t_emb + y_emb

        # Refine embeddings
        predicted_embeddings = self.refine_embeddings(real_embeddings, initial_predicted_embeddings, c, t)

        # Project back to token dimension and unpatchify
        predicted_tokens = self.output_proj(predicted_embeddings)  # [B, S, token_embed_dim]
        x_0_pred = self.unpatchify(predicted_tokens)  # [B, C, H, W]

        return x_0_pred

    def forward(self, x_start, labels):
        """
        Args:
            x_start: (B, C, H, W) - Clean image
            labels: (B,) - Class labels
        """
        bsz = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.diffusion.num_timesteps, (bsz,), device=device)

        # Compute noisy image x_t
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        # Predict x_0 from x_t
        x_0_pred = self._forward_impl(x_t, t, labels, x_start=x_start)

        # Compute the loss (assuming the diffusion loss expects x_0 prediction)
        loss_dict = self.diffusion.training_losses(
            model=self._forward_impl,
            x_start=x_start,
            t=t,
            model_kwargs={"labels": labels, "x_start": x_start},
            noise=noise,
        )
        return loss_dict["loss"].mean()

    def forward_with_cfg(self, x, t, labels, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        t_combined = torch.cat([t[:len(t) // 2], t[:len(t) // 2]], dim=0)

        # Conditional and unconditional labels
        labels_cond = labels[:len(labels) // 2]
        labels_uncond = torch.full_like(labels_cond, self.y_embedder.num_embeddings - 1)
        labels_combined = torch.cat([labels_cond, labels_uncond], dim=0)

        # Forward pass
        model_output = self._forward_impl(combined, t_combined, labels_combined)

        # Split conditional and unconditional outputs
        model_output_cond, model_output_uncond = torch.split(model_output, len(model_output) // 2, dim=0)

        # Apply CFG formula (assuming output is x_0)
        guided_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
        return guided_output

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, labels=None, temperature=1.0, progress=False):
        """
        Sample images using the diffusion process.
        """
        device = next(self.parameters()).device
        shape = (bsz, self.vae_embed_dim, self.img_size // self.vae_stride, self.img_size // self.vae_stride)
        img = torch.randn(*shape, device=device)

        if labels is None:
            labels = torch.randint(0, self.y_embedder.num_embeddings - 1, (bsz,), device=device)

        def model_fn(x, ts):
            if cfg > 1.0:
                return self.forward_with_cfg(x, ts, labels, cfg_scale=cfg)
            else:
                return self._forward_impl(x, ts, labels)

        samples = self.diffusion.p_sample_loop(
            model_fn,
            shape,
            noise=img,
            device=device,
            progress=progress
        )
        return samples
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h
    
class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class EnergyMLP(nn.Module):
    """Energy-based MLP with AdaLN, conditioned on z"""
    def __init__(self, token_embed_dim, z_dim, hidden_dim=1024, depth=6, mcmc_num_steps=2, 
                 alpha=.01, langevin_noise_std=0, reconstruction_coeff=1.0, grad_checkpointing=False):
        super().__init__()
        self.token_embed_dim = token_embed_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.mcmc_num_steps = mcmc_num_steps
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=True)  # Step size for updates, set to learnable
        self.langevin_noise_std = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reconstruction_coeff = reconstruction_coeff
        self.grad_checkpointing = grad_checkpointing

        # Project predicted embeddings to hidden space
        self.input_proj = nn.Linear(token_embed_dim, hidden_dim)
        # Project condition z to hidden space
        self.cond_proj = nn.Linear(z_dim, hidden_dim)
        # AdaLN residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, 1)

    def compute_energy(self, predicted_embeddings, z):
        """Compute image-level energy.

        Args:
            predicted_embeddings (Tensor): shape (B, S, token_dim)
            z (Tensor):                   shape (B, S, z_dim)

        Returns:
            Tensor: (B, 1) energy for each image.
        """
        B, S, _ = predicted_embeddings.shape

        # project to hidden
        x = self.input_proj(predicted_embeddings)   # (B, S, hidden)
        c = self.cond_proj(z)                       # (B, S, hidden)

        # flatten tokens so Residual blocks are applied per-token (parameters shared)
        x = x.reshape(B * S, self.hidden_dim)
        c = c.reshape(B * S, self.hidden_dim)

        for block in self.res_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, c)
            else:
                x = block(x, c)

        # reshape back and aggregate across tokens to obtain image level representation
        x = x.view(B, S, self.hidden_dim)
        c = c.view(B, S, self.hidden_dim)

        # simple mean pool over spatial tokens
        x_agg = x.mean(dim=1)  # (B, hidden)
        c_agg = c.mean(dim=1)  # (B, hidden)

        energy = self.final_layer(x_agg, c_agg)  # (B, 1)
        return energy

    def forward(self, z, real_embeddings_input):
        """Forward pass with per-token conditioning.

        Inputs
        z: (B, S, decoder_dim)
        real_embeddings_input: (B, S, token_dim)
        """
        B, S, _ = real_embeddings_input.shape

        predicted_embeddings_list = []
        predicted_energies_list = []

        alpha = torch.clamp(self.alpha, min=1e-4)

        # init noise for predicted embeddings (per image)
        predicted_embeddings = torch.randn(B, S, self.token_embed_dim, device=z.device)

        with torch.enable_grad():
            for _ in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()

                # Compute energy for each image
                energy = self.compute_energy(predicted_embeddings, z)  # (B,1)

                # store for logging
                predicted_energies_list.append(energy)

                # grad w.r.t predicted embeddings
                grad = torch.autograd.grad(energy.sum(), predicted_embeddings, create_graph=True, retain_graph=True)[0]

                # update
                predicted_embeddings = predicted_embeddings - alpha * grad

                # store prediction
                predicted_embeddings_list.append(predicted_embeddings)

        return predicted_embeddings_list, predicted_energies_list

    def sample(self, z_cond, z_uncond=None, cfg=1.0, temperature=1.0, init_embeddings=None, init_embeddings_uncond=None):
            """Sample full-image embeddings with optional classifier-free guidance.

            Args:
                z_cond (Tensor): (B, S, z_dim)
                z_uncond (Tensor|None): same shape as z_cond for unconditional branch.
                cfg (float): guidance scale.
                temperature (float): currently only scales initial noise.
                init_embeddings (Tensor|None): optional starting embeddings for conditional branch (B,S,D).
                init_embeddings_uncond (Tensor|None): optional starting embeddings for unconditional branch.
            Returns:
                Tensor: predicted embeddings for the conditional branch (B,S,D).
            """

            B, S, _ = z_cond.shape

            # initialise embeddings
            if init_embeddings is None:
                pred_cond = torch.randn(B, S, self.token_embed_dim, device=z_cond.device) * temperature
            else:
                pred_cond = init_embeddings.clone().to(z_cond.device)

            if z_uncond is not None:
                if init_embeddings_uncond is None:
                    pred_uncond = torch.randn_like(pred_cond) * temperature
                else:
                    pred_uncond = init_embeddings_uncond.clone().to(z_cond.device)

            alpha = torch.clamp(self.alpha, min=1e-4)

            with torch.enable_grad():
                for _ in range(self.mcmc_num_steps):
                    # conditional branch
                    pred_cond = pred_cond.detach().requires_grad_()
                    energy_cond = self.compute_energy(pred_cond, z_cond)
                    grad_cond = torch.autograd.grad(energy_cond.sum(), pred_cond)[0]

                    if z_uncond is not None and cfg > 1.0:
                        # unconditional branch
                        pred_uncond = pred_uncond.detach().requires_grad_()
                        energy_uncond = self.compute_energy(pred_uncond, z_uncond)
                        grad_uncond = torch.autograd.grad(energy_uncond.sum(), pred_uncond)[0]

                        grad = (1 + cfg) * grad_cond - cfg * grad_uncond
                        pred_cond = pred_cond - alpha * grad
                        pred_uncond = pred_uncond - alpha * grad_uncond  # optional update
                    else:
                        pred_cond = pred_cond - alpha * grad_cond

            return pred_cond.detach()

    def compute_loss(self, predicted_embeddings_list, real_embeddings_input):
        """Compute reconstruction loss"""
        final_predicted = predicted_embeddings_list[-1]
        loss = ((final_predicted - real_embeddings_input) ** 2).mean()
        return loss * self.reconstruction_coeff
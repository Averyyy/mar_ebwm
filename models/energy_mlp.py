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
        # self.langevin_noise_std = nn.Parameter(torch.tensor(float(langevin_noise_std)), requires_grad=False)  # Noise scale
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
        """Compute token-level energy.

        predicted_embeddings: (N, token_dim)
        z                : (N, decoder_dim)
        """
        x = self.input_proj(predicted_embeddings)   # (N, hidden)
        c = self.cond_proj(z)                      # (N, hidden)

        for block in self.res_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, c)
            else:
                x = block(x, c)
        energy = self.final_layer(x, c)                # (N,1)
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
        
        z_flat = z.reshape(B * S, -1)  # (B*S, decoder_dim)

        alpha = torch.clamp(self.alpha, min=1e-4)

        # init noise for predicted embeddings
        predicted_embeddings = torch.randn(B * S, self.token_embed_dim, device=z.device)

        with torch.enable_grad():
            for _ in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()

                # Compute energy
                energy = self.compute_energy(predicted_embeddings, z_flat)  # (B*S,1)

                # store for logging (reshape back)
                predicted_energies_list.append(energy.view(B, S, -1))

                # grad
                grad = torch.autograd.grad(energy.sum(), predicted_embeddings, create_graph=True, retain_graph=True)[0]

                # update
                predicted_embeddings = predicted_embeddings - alpha * grad

                # store prediction (reshape back)
                predicted_embeddings_list.append(predicted_embeddings.view(B, S, self.token_embed_dim))

        return predicted_embeddings_list, predicted_energies_list

    def sample(self, z_cond, z_uncond=None, cfg=1.0, temperature=1.0):
            """Sample embeddings with optional CFG"""
            N, _ = z_cond.shape  # z_cond is [N, z_dim]
            predicted_embeddings = torch.randn(N, self.token_embed_dim, device=z_cond.device)
            with torch.enable_grad():
                for _ in range(self.mcmc_num_steps):
                    predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                    energy_cond = self.compute_energy(predicted_embeddings, z_cond)
                    grad_cond = torch.autograd.grad(energy_cond.sum(), predicted_embeddings)[0]
                    if z_uncond is not None and cfg > 1.0:
                        energy_uncond = self.compute_energy(predicted_embeddings, z_uncond)
                        grad_uncond = torch.autograd.grad(energy_uncond.sum(), predicted_embeddings)[0]
                        grad = (1 + cfg) * grad_cond - cfg * grad_uncond
                    else:
                        grad = grad_cond
                    predicted_embeddings = predicted_embeddings - self.alpha * grad
            return predicted_embeddings.detach()

    def compute_loss(self, predicted_embeddings_list, real_embeddings_input):
        """Compute reconstruction loss"""
        final_predicted = predicted_embeddings_list[-1]
        loss = ((final_predicted - real_embeddings_input) ** 2).mean()
        return loss * self.reconstruction_coeff
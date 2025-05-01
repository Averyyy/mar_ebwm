import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def modulate(x, shift, scale):
    """Adaptive Layer Normalization modulation"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNResBlock(nn.Module):
    """AdaLN Residual Block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)  # Outputs shift, scale, gate
        )

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        h = self.mlp(h)
        return x + gate.unsqueeze(1) * h

class EnergyMLP(nn.Module):
    """Energy-based MLP with AdaLN, conditioned on z"""
    def __init__(self, token_embed_dim, z_dim, hidden_dim=1024, depth=3, mcmc_num_steps=2, 
                 alpha=.01, langevin_noise_std=0, reconstruction_coeff=1.0, grad_checkpointing=False):
        super().__init__()
        self.token_embed_dim = token_embed_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.mcmc_num_steps = mcmc_num_steps
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=True)  # Step size for updates, set to learnable
        self.langevin_noise_std = nn.Parameter(torch.tensor(float(langevin_noise_std)), requires_grad=False)  # Noise scale
        self.reconstruction_coeff = reconstruction_coeff
        self.grad_checkpointing = grad_checkpointing

        # Project predicted embeddings to hidden space
        self.input_proj = nn.Linear(token_embed_dim, hidden_dim)
        # Project condition z to hidden space
        self.cond_proj = nn.Linear(z_dim, hidden_dim)
        # AdaLN residual blocks
        self.res_blocks = nn.ModuleList([AdaLNResBlock(hidden_dim) for _ in range(depth)])
        # Final layer to output energy scalar
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, 1)
        )

    def compute_energy(self, predicted_embeddings, z):
        """Compute energy conditioned on z"""
        x = self.input_proj(predicted_embeddings)  # [B, S, hidden_dim]  |  [N, hidden_dim]
        c = self.cond_proj(z)  # [B, S, hidden_dim]  |  [N, hidden_dim]
        for block in self.res_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, c)
            else:
                x = block(x, c)
        energy = self.final_layer(x)  # [B, S, 1]  |  [N, 1]
        return energy

    def forward(self, z, real_embeddings_input):
        """Forward pass: Optimize predicted embeddings from noise"""
        B, S, D = real_embeddings_input.shape
        predicted_embeddings_list = []
        predicted_energies_list = []
        
        alpha = torch.clamp(self.alpha, min=0.0001)
        # langevin_noise_std = torch.clamp(self.langevin_noise_std, min=0.000001)

        # Initialize predicted embeddings as pure noise
        predicted_embeddings = torch.randn(B, S, self.token_embed_dim, device=z.device)

        with torch.enable_grad():
            for _ in range(self.mcmc_num_steps):
                predicted_embeddings = predicted_embeddings.detach().requires_grad_()
                # Add Langevin dynamics noise
                # noise = torch.randn_like(predicted_embeddings.detach()) * langevin_noise_std #TODO add this conditionally if its not 0 and set langevin_noise_std to 0 for now
                # predicted_embeddings = predicted_embeddings + noise
                # Compute energy
                energy = self.compute_energy(predicted_embeddings, z)
                predicted_energies_list.append(energy)
                # Compute gradient of energy
                grad = torch.autograd.grad(energy.sum(), predicted_embeddings, create_graph=True, retain_graph=True)[0]
                # Update predicted embeddings
                predicted_embeddings = predicted_embeddings - alpha * grad
                predicted_embeddings_list.append(predicted_embeddings)

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
                    # noise = torch.randn_like(predicted_embeddings) * temperature
                    # predicted_embeddings = predicted_embeddings - self.alpha * grad + noise
                    predicted_embeddings = predicted_embeddings - self.alpha * grad
            return predicted_embeddings.detach()

    def compute_loss(self, predicted_embeddings_list, real_embeddings_input):
        """Compute reconstruction loss"""
        final_predicted = predicted_embeddings_list[-1]
        loss = ((final_predicted - real_embeddings_input) ** 2).mean()
        return loss * self.reconstruction_coeff
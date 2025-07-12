import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ebt.ebt_adaln import EBTAdaLN
from models.ebt.model_utils import EBTModelArgs


def extract(a, t, x_shape):
    """Extract values from a 1-D tensor for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule for diffusion."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DEBT_EBM(nn.Module):
    """Energy-Based Model for autoregressive DEBT with diffusion"""
    
    def __init__(self, 
                 seq_len, 
                 token_embed_dim, 
                 embed_dim, 
                 depth, 
                 num_heads, 
                 class_num):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed_dim = token_embed_dim
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Linear(token_embed_dim, embed_dim)
        
        self.y_embedder = nn.Embedding(class_num + 1, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 2, embed_dim))
        
        # EBT
        t_args = EBTModelArgs(
            dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            max_seq_len=seq_len + 2,
            final_output_dim=1,  # output energy
        )
        self.transformer = EBTAdaLN(params=t_args, max_mcmc_steps=10)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def get_time_embedding(self, timesteps):
        """Sinusoidal time embeddings"""
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)
    
    def forward(self, inp_tokens, opt_out_tokens, t, cond=None):
        """
        Compute energy for diffusion.
        
        Args:
            inp_tokens: [B, seq_len, token_embed_dim] - real sequence (already formatted with start)
            opt_out_tokens: [B, seq_len, token_embed_dim] - pred sequence (already formatted with end)
            t: [B] - diffusion timesteps
            cond: [B] - class conditioning
            
        Returns:
            energy: [B, 1] - scalar energy
        """
        B = inp_tokens.size(0)
        
        real_embeddings = self.input_proj(inp_tokens)  # [B, S, D]
        pred_embeddings = self.input_proj(opt_out_tokens)  # [B, S, D]

        real_embeddings = real_embeddings + self.pos_embed[:, :real_embeddings.size(1), :]
        pred_embeddings = pred_embeddings + self.pos_embed[:, 1:1 + pred_embeddings.size(1), :]

        all_embeddings = torch.cat([real_embeddings, pred_embeddings], dim=1)  # [B, 2*(S+1), D]
        
        time_vec = self.get_time_embedding(t)  # [B, D]
        # c = time_embedding + class_embedding (if given)
        if cond is not None:
            cond_embeddings = self.y_embedder(cond) + time_vec
        else:
            cond_embeddings = time_vec
        
        # Get per-token energy from EBT
        energy_per_token = self.transformer(
            all_embeddings, 
            start_pos=0, 
            mcmc_step=0, 
            c=cond_embeddings
        )  # [B, seq_len, 1]
        
        return energy_per_token  # [B, seq_len, 1]


class DEBTDiffusionWrapper(nn.Module):
    """Converts EBM energy gradients to diffusion noise predictions"""
    
    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm
    
    def forward(self, inp_tokens, opt_out_tokens, t, cond=None, *, return_energy: bool=False, return_both: bool=False):
        """        
        Returns:
            grad: [B, opt_len, D] - energy gradients (score)
        """
        opt_out_tokens = opt_out_tokens.detach().requires_grad_(True)

        inp_len = inp_tokens.size(1)
        opt_len = opt_out_tokens.size(1)


        if opt_len < inp_len:
            prefix = inp_tokens[:, 1:inp_len].detach()  # (B, inp_len-1, D)
            padded_opt_tokens = torch.cat([prefix, opt_out_tokens], dim=1)  # (B, inp_len, D)
            energy_per_token = self.ebm(inp_tokens, padded_opt_tokens, t, cond)  # [B, L, 1]
            # Only use energy corresponding to the last token
            energy_scalar = energy_per_token[:, -1, 0]
        else:
            # Training / full sequence path
            energy_scalar = self.ebm(inp_tokens, opt_out_tokens, t, cond).sum(dim=1)

        if return_energy and not return_both:
            return energy_scalar.unsqueeze(1)

        grad = torch.autograd.grad(
            energy_scalar.sum(),
            opt_out_tokens,
            create_graph=True
        )[0]

        if return_both:
            return energy_scalar.unsqueeze(1), grad
        else:
            return grad

class SimpleDiffusion(nn.Module):
    """Simple diffusion process for IRED token generation"""
    
    def __init__(self, model, timesteps=10, inner_steps: int = 0):
        super().__init__()
        self.model = model  # DEBTDiffusionWrapper
        self.timesteps = timesteps
        self.inner_steps = inner_steps
        self.use_innerloop_opt = inner_steps > 0
        
        # Cosine schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod).float())

        # Inner-loop step size: β_t * sqrt(1 / (1-ᾱ_t))
        opt_step_size = betas * torch.sqrt(1.0 / torch.clamp(1.0 - alphas_cumprod, min=1e-8))
        self.register_buffer('opt_step_size', opt_step_size.float())

    def opt_step(self, inp_tokens, x, t, cond):
        """opt step to refine energy landscape according to IRED"""
        if self.inner_steps == 0:
            return x

        x_cur = x.detach()
        for _ in range(self.inner_steps):
            x_cur = x_cur.detach().requires_grad_(True)
            energy, grad = self.model(inp_tokens, x_cur, t, cond, return_both=True)

            step_size = extract(self.opt_step_size, t, grad.shape)
            x_new = x_cur - step_size * grad

            if not self.training:
                energy_new = self.model(inp_tokens, x_new, t, cond, return_energy=True)
                accept = (energy_new <= energy).float().reshape(-1, *([1]*(x_cur.dim()-1)))
                x_cur = accept * x_new + (1 - accept) * x_cur
            else:
                x_cur = x_new

        return x_cur.detach()
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, inp_tokens, x, t, cond):
        """single denoising step using energy gradients"""
        pred_noise = self.model(inp_tokens, x, t, cond)

        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / (sqrt_alphas_cumprod_t + 1e-8)

        #    x_{t-1} = sqrt(alpha_cumprod_{t-1}) * pred_x0 + sqrt(1 - alpha_cumprod_{t-1}) * pred_noise
        t_prev = torch.clamp(t - 1, min=0)
        sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod, t_prev, x.shape)
        sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod, t_prev, x.shape)

        x_new = sqrt_alphas_cumprod_prev * pred_x0 + sqrt_one_minus_alphas_cumprod_prev * pred_noise

        # inner-loop refinement
        if self.use_innerloop_opt:
            x_new = self.opt_step(inp_tokens, x_new, t_prev, cond)

        # Clear intermediate tensors to save memory
        del pred_noise, sqrt_one_minus_alphas_cumprod_t, sqrt_alphas_cumprod_t, \
            pred_x0, sqrt_alphas_cumprod_prev, sqrt_one_minus_alphas_cumprod_prev
        return x_new
    
    def sample(self, inp_tokens, shape, cond):
        """Generate tokens using reverse diffusion"""
        device = inp_tokens.device
        batch_size = inp_tokens.size(0)
        
        x = torch.randn(batch_size, *shape, device=device)
        
        # diffusion steps
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(inp_tokens, x, t, cond)
        
        return x


class DEBTDiffusion(nn.Module):
    """DEBT with IRED Diffusion"""
    
    def __init__(self,
                 img_size=64,
                 vae_stride=8,
                 patch_size=1,
                 embed_dim=768,
                 depth=24,
                 num_heads=12,
                 class_num=1000,
                 diffusion_timesteps=10,
                 inner_steps=0):
        super().__init__()
        
        # Image parameters
        self.img_size = img_size
        self.vae_stride = vae_stride  
        self.patch_size = patch_size

        # 保存类别数，供采样默认使用
        self.class_num = class_num
        
        # Token parameters
        latent_size = img_size // vae_stride
        self.seq_len = (latent_size // patch_size) ** 2
        self.token_embed_dim = 16 * (patch_size ** 2)

        # Learnable special tokens in token space
        self.start_token = nn.Parameter(torch.randn(1, 1, self.token_embed_dim))
        self.end_token = nn.Parameter(torch.randn(1, 1, self.token_embed_dim))
        
        # EBM for energy computation
        self.ebm = DEBT_EBM(
            seq_len=self.seq_len,
            token_embed_dim=self.token_embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            class_num=class_num,
        )
        
        # Diffusion wrapper and process
        self.diffusion_wrapper = DEBTDiffusionWrapper(self.ebm)
        self.diffusion = SimpleDiffusion(self.diffusion_wrapper, timesteps=diffusion_timesteps, inner_steps=inner_steps)
    
    def patchify(self, x):
        """Convert image to token sequence"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, -1, C * self.patch_size ** 2)
    
    def unpatchify(self, x):
        """Convert token sequence to image"""
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        C = D // (self.patch_size ** 2)
        x = x.view(B, H, W, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, H * self.patch_size, W * self.patch_size)
    
    def forward(self, x_start, labels):
        """Training forward pass with DEBT structure"""

        gt_tokens = self.patchify(x_start)  # [B, seq_len, D]
        B = gt_tokens.shape[0]
        device = x_start.device
        
        start_token = self.start_token.expand(B, -1, -1)
        end_token = self.end_token.expand(B, -1, -1)
        
        # real_seq: [<start>] + all ground-truth tokens
        inp_tokens = torch.cat([start_token, gt_tokens], dim=1)  # [B, S+1, D]
        
        # pred_seq: all ground-truth tokens + [<end>]
        target_tokens = torch.cat([gt_tokens, end_token], dim=1)  # [B, S+1, D]
        
        # add noise to target
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=device)
        noise = torch.randn_like(target_tokens)
        noisy_target = self.diffusion.q_sample(target_tokens, t, noise)
        
        pred_noise = self.diffusion_wrapper(inp_tokens, noisy_target, t, labels)
        
        # Diffusion loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def sample_tokens(self, bsz, labels=None, **kwargs):
        """Autoregressive token generation with IRED diffusion"""
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.class_num, (bsz,), device=device)
        else:
            assert labels.max() < self.class_num, f"Label id {labels.max().item()} exceeds class_num ({self.class_num})"
        
        num_tokens_to_generate = self.seq_len
        
        was_training = self.training
        self.eval()
        
        start_token = self.start_token.expand(bsz, -1, -1)
        generated_tokens = []
        
        # ar generation
        for step in range(num_tokens_to_generate):
            if step % 4 == 0:
                torch.cuda.empty_cache()
            
            if step == 0:
                real_seq = start_token  # [B, 1, D]
            else:
                real_seq = torch.cat([start_token] + generated_tokens, dim=1)  # [B, step+1, D]
            
            # Generate next token using diffusion
            with torch.no_grad():
                x = torch.randn(bsz, 1, self.token_embed_dim, device=device)
                
                for t_idx in reversed(range(self.diffusion.timesteps)):
                    t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
                    
                    with torch.enable_grad():
                        x = self.diffusion.p_sample(real_seq, x, t, labels)
                
                next_token = x.detach()  # [B, 1, D]
            
            generated_tokens.append(next_token)
        
        final_tokens = torch.cat(generated_tokens, dim=1)  # [B, seq_len, D]
        
        result = self.unpatchify(final_tokens)
        
        if was_training:
            self.train()
            
        return result


def debt_diffusion_2xs(**kwargs):
    defaults = dict(embed_dim=384, depth=6, num_heads=6, diffusion_timesteps=50)
    defaults.update(kwargs)
    return DEBTDiffusion(**defaults)

def debt_diffusion_base(**kwargs):
    defaults = dict(embed_dim=768, depth=24, num_heads=12, diffusion_timesteps=500)
    defaults.update(kwargs)
    return DEBTDiffusion(**defaults)

def debt_diffusion_large(**kwargs):
    defaults = dict(embed_dim=1024, depth=32, num_heads=16, diffusion_timesteps=500)
    defaults.update(kwargs)
    return DEBTDiffusion(**defaults)

def debt_diffusion_huge(**kwargs):
    defaults = dict(embed_dim=1280, depth=40, num_heads=16, diffusion_timesteps=1000)
    defaults.update(kwargs)
    return DEBTDiffusion(**defaults)


if __name__ == "__main__":
    print("=== Testing DEBT Diffusion ===")
    
    model = debt_diffusion_2xs(img_size=64, class_num=10)
    
    x = torch.randn(2, 16, 8, 8)
    labels = torch.tensor([0, 1])
    
    print("Training test:")
    model.train()
    loss = model(x, labels)
    print(f"Loss: {loss.item():.4f}")
    print("✓ Training passed")
    
    # Test sampling
    print("\nSampling test:")
    model.eval()
    sample = model.sample_tokens(bsz=1, num_iter=4, labels=torch.tensor([0]))
    print(f"Sample shape: {sample.shape}")
    print("✓ Sampling passed")
    
    print("\n✓ Clean DEBT Diffusion implementation working!")
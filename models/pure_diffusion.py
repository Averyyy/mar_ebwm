import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional
import torch.nn.functional as F

from models.dit.dit import DiT, DiT_models
from diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType


class PureDiffusion(nn.Module):
    """
    Pure diffusion model that takes VAE encoded latents as input.
    Compatible with MAR training pipeline but uses purely diffusion-based generation.
    """
    
    def __init__(
        self,
        # Image/latent parameters
        img_size=256,
        vae_stride=16, 
        vae_embed_dim=16,
        
        # Model architecture
        dit_model="DiT-B/2",
        
        # Diffusion parameters
        num_diffusion_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        
        # Class conditioning
        class_num=1000,
        class_dropout_prob=0.1,
        
        # Energy mode
        use_energy=False,
        use_innerloop_opt=False,
        
        # Compatibility parameters
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = vae_embed_dim
        self.num_classes = class_num
        self.use_energy = use_energy
        self.use_innerloop_opt = use_innerloop_opt
        
        # Initialize DiT model
        # Convert latent tokens to spatial format for DiT
        latent_size = img_size // vae_stride  # Size of latent space
        dit_kwargs = {
            'input_size': latent_size,
            'in_channels': vae_embed_dim,
            'num_classes': class_num,
            'class_dropout_prob': class_dropout_prob,
            'learn_sigma': False,  # Don't learn variance, only predict noise
            'use_energy': use_energy,
        }
        
        if dit_model in DiT_models:
            self.dit = DiT_models[dit_model](**dit_kwargs)
        else:
            raise ValueError(f"Unknown DiT model: {dit_model}")
        
        # Initialize diffusion process
        betas = get_named_beta_schedule(beta_schedule, num_diffusion_timesteps)
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
            
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
        )
        
        # For compatibility with MAR pipeline
        self.use_energy_loss = False  # This is a pure diffusion model
    
    def forward(self, x, labels):
        """
        Training forward pass. Compatible with MAR training pipeline.
        
        Args:
            x: VAE encoded latent tensor [B, C, H, W]
            labels: Class labels [B]
        
        Returns:
            Diffusion loss
        """
        # Sample random timesteps
        bsz = x.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (bsz,), device=x.device)
        
        # Compute diffusion loss
        loss = self.diffusion.training_losses(
            model=self.dit,
            x_start=x,
            t=t,
            model_kwargs={"y": labels}
        )
        
        return loss["loss"].mean()
    
    def opt_step(self, x, t, labels, step=5, step_size=0.01):
        """
        Optimization step for energy diffusion following IRED approach.
        Performs gradient descent on energy landscape to refine samples.
        
        Args:
            x: Current sample [B, C, H, W]
            t: Timestep [B]
            labels: Class labels [B]
            step: Number of optimization steps
            step_size: Step size for gradient descent
        
        Returns:
            Optimized sample
        """
        if not self.use_energy:
            return x
            
        with torch.enable_grad():
            x_opt = x.clone().requires_grad_(True)
            
            for i in range(step):
                # Get energy and gradients
                energy, gradients = self.dit(x_opt, t, labels, return_both=True)
                
                # Gradient descent step to minimize energy
                x_new = x_opt - step_size * gradients.float()
                
                # Clamp values to reasonable range (similar to IRED)
                # Use the diffusion forward process scaling
                alpha_cumprod = torch.from_numpy(self.diffusion.alphas_cumprod).float().to(t.device)[t]
                max_val = torch.sqrt(alpha_cumprod).view(-1, 1, 1, 1) * 2.0
                x_new = torch.clamp(x_new, -max_val, max_val)
                
                # Check if energy decreased, if not, reject the step
                energy_new = self.dit(x_new, t, labels, return_energy=True)
                bad_step = (energy_new > energy).squeeze()
                
                # Keep old values where energy increased
                if bad_step.any():
                    x_new[bad_step] = x_opt[bad_step]
                
                x_opt = x_new.detach().requires_grad_(True)
        
        return x_opt.detach()
    
    def energy_aware_p_sample_loop(self, shape, labels, progress=False):
        """
        Modified p_sample_loop that includes energy optimization steps.
        Based on IRED's approach of calling opt_step during sampling.
        """
        device = next(self.dit.parameters()).device
        bsz = labels.shape[0]
        
        # Initialize with noise
        x = torch.randn(bsz, *shape, device=device)
        
        timesteps = list(range(self.diffusion.num_timesteps))[::-1]
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((bsz,), t_val, device=device, dtype=torch.long)
            
            # Standard diffusion sampling step
            with torch.no_grad():
                out = self.diffusion.p_sample(
                    self.dit,
                    x,
                    t, 
                    model_kwargs={"y": labels}
                )
                x = out["sample"]
            
            # Apply energy optimization if enabled
            if self.use_energy and self.use_innerloop_opt:
                # More optimization steps for later timesteps (following IRED pattern)
                opt_steps = 5 if t_val > 1 else 2
                x = self.opt_step(x, t, labels, step=opt_steps, step_size=0.01)
                
        return x
    
    def sample_tokens(
        self, 
        bsz, 
        num_iter=None,  # Ignored for diffusion
        cfg=1.0, 
        cfg_schedule="linear",  # Ignored for diffusion
        labels=None, 
        temperature=1.0,  # Ignored for diffusion
        progress=False,
        gt_prefix_tokens=None,  # Ignored for diffusion
        **kwargs
    ):
        """
        Generate samples using pure diffusion. Compatible with MAR sample_tokens interface.
        
        Args:
            bsz: Batch size
            num_iter: Ignored (diffusion uses fixed timesteps)
            cfg: Classifier-free guidance scale
            cfg_schedule: Ignored for diffusion
            labels: Class labels [B]
            temperature: Ignored for diffusion
            progress: Whether to show progress
            gt_prefix_tokens: Ignored for diffusion
            
        Returns:
            Generated latent tokens [B, C, H, W]
        """
        device = next(self.dit.parameters()).device
        
        if labels is None:
            labels = torch.randint(0, self.num_classes, (bsz,), device=device)
        
        # Sample noise
        latent_size = self.img_size // self.vae_stride
        shape = (bsz, self.vae_embed_dim, latent_size, latent_size)
        
        if cfg != 1.0:
            # Classifier-free guidance sampling
            # Use the DiT's built-in CFG
            def cfg_model(x, t, **kwargs):
                return self.dit.forward_with_cfg(x, t, kwargs.get("y"), cfg)
            
            samples = self.diffusion.p_sample_loop(
                model=cfg_model,
                shape=shape,
                clip_denoised=True,
                model_kwargs={"y": labels},
                cond_fn=None,
                device=device,
                progress=progress,
            )
        else:
            # Check if we need energy-aware sampling
            if self.use_energy and self.use_innerloop_opt:
                # Use our custom energy-aware sampling loop
                samples = self.energy_aware_p_sample_loop(
                    shape=(self.vae_embed_dim, latent_size, latent_size),
                    labels=labels,
                    progress=progress
                )
            else:
                # Regular sampling
                samples = self.diffusion.p_sample_loop(
                    model=self.dit,
                    shape=shape,
                    clip_denoised=True,
                    model_kwargs={"y": labels},
                    cond_fn=None,
                    device=device,
                    progress=progress,
                )
        return samples


def pure_diffusion_small(**kwargs):
    """Pure diffusion model with DiT-S backbone"""
    return PureDiffusion(dit_model="DiT-S/1", **kwargs)


def pure_diffusion_base(**kwargs):
    """Pure diffusion model with DiT-B backbone"""
    return PureDiffusion(dit_model="DiT-B/2", **kwargs)


def pure_diffusion_large(**kwargs):
    """Pure diffusion model with DiT-L backbone"""
    return PureDiffusion(dit_model="DiT-L/2", **kwargs)


def pure_diffusion_xlarge(**kwargs):
    """Pure diffusion model with DiT-XL backbone"""
    return PureDiffusion(dit_model="DiT-XL/2", **kwargs)
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit.dit import DiT, DiT_models

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
        
        # Class conditioning
        class_num=1000,
        class_dropout_prob=0.1,
        
        # Sampling parameters
        num_sampling_steps=250,
        
        # Energy mode
        use_energy=False,
        use_innerloop_opt=False,
        always_accept_opt_steps=False,
        supervise_energy_landscape=False,
        learnable_mcmc_step_size=False,
        mcmc_step_size=0.01,
        linear_then_mean=False,
        contrasive_loss_scale=0.05,
        mcmc_refinement_loss_scale=0.1,
        log_energy_accept_rate=False,
        energy_gradient_multiplier=1.0,
        
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.num_sampling_steps = num_sampling_steps
        self.vae_stride = vae_stride
        self.vae_embed_dim = vae_embed_dim
        self.num_classes = class_num
        self.use_energy = use_energy
        self.use_innerloop_opt = use_innerloop_opt
        self.always_accept_opt_steps = always_accept_opt_steps
        self.supervise_energy_landscape = supervise_energy_landscape
        self.contrasive_loss_scale = contrasive_loss_scale
        self.mcmc_refinement_loss_scale = mcmc_refinement_loss_scale
        self.log_energy_accept_rate = log_energy_accept_rate
        self.learnable_mcmc_step_size = learnable_mcmc_step_size
        
        # Create alpha parameter - learnable if specified, otherwise fixed
        if learnable_mcmc_step_size:
            self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=True)
        else:
            self.register_buffer('alpha', torch.tensor(float(mcmc_step_size)))
        
        # Initialize DiT model
        latent_size = img_size // vae_stride  # Size of latent space
        dit_kwargs = {
            'input_size': latent_size,
            'in_channels': vae_embed_dim,
            'num_classes': class_num,
            'class_dropout_prob': class_dropout_prob,
            'learn_sigma': False,
            'use_energy': use_energy,
            'linear_then_mean': linear_then_mean,
            'energy_gradient_multiplier': energy_gradient_multiplier,
        }
        
        if dit_model in DiT_models:
            self.dit = DiT_models[dit_model](**dit_kwargs)
        else:
            raise ValueError(f"Unknown DiT model: {dit_model}")
        
        from diffusion import create_diffusion
        self.train_diffusion = create_diffusion(
            timestep_respacing="",  # Full timesteps during training
            noise_schedule=beta_schedule,
            sigma_small=True,
            learn_sigma=False,
            diffusion_steps=num_diffusion_timesteps,
        )
        
        self.gen_diffusion = create_diffusion(
            timestep_respacing=str(num_sampling_steps),
            noise_schedule=beta_schedule,
            sigma_small=True,
            learn_sigma=False,
            diffusion_steps=num_diffusion_timesteps,
        )
        
    
    def compute_contrastive_loss(self, x, t, labels):
        """
        Compute contrastive energy loss using positive and negative samples.
        
        Args:
            x: Clean samples [B, C, H, W]
            t: Timesteps [B]
            labels: Class labels [B]
            
        Returns:
            Contrastive energy loss tensor
        """
        # Add noise to create noisy version
        noise = torch.randn_like(x)
        
        # Generate negative samples through energy optimization
        x_neg_start = x + 3.0 * torch.randn_like(x)  # Start from perturbed version
        x_neg_noisy = self.train_diffusion.q_sample(x_start=x_neg_start, t=t, noise=noise)
        
        # Optimize negative samples using energy landscape with detached alpha (no gradient)
        alpha_detached = self.alpha.detach() if hasattr(self, 'alpha') and self.alpha.requires_grad else 0.01
        opt_result = self.opt_step(x_neg_noisy, t, labels, step=None, step_size=alpha_detached, keep_grad=False)
        
        # Handle potential tuple return from opt_step (when logging accept rates)
        if isinstance(opt_result, tuple):
            x_neg_opt = opt_result[0]  # Extract tensor from tuple
        else:
            x_neg_opt = opt_result
        
        # Predict x0 from optimized negative samples
        alpha_cumprod = torch.from_numpy(self.train_diffusion.alphas_cumprod).float().to(t.device)[t]
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1, 1)
        
        # Predict x0 from the optimized noisy sample (reverse of q_sample)
        x_neg_pred = (x_neg_opt - sqrt_one_minus_alpha_cumprod * torch.zeros_like(x_neg_opt)) / sqrt_alpha_cumprod
        x_neg_pred = torch.clamp(x_neg_pred, -2, 2)
        
        # Create new noisy versions for energy computation
        x_pos_noisy = self.train_diffusion.q_sample(x_start=x, t=t, noise=noise)
        x_neg_noisy_final = self.train_diffusion.q_sample(x_start=x_neg_pred, t=t, noise=noise)
        
        # Compute energy for both positive and negative samples
        x_concat = torch.cat([x_pos_noisy, x_neg_noisy_final], dim=0)
        t_concat = torch.cat([t, t], dim=0)
        labels_concat = torch.cat([labels, labels], dim=0)
        
        energy = self.dit(x_concat, t_concat, labels_concat, return_energy=True)
        energy_pos, energy_neg = torch.chunk(energy, 2, dim=0)
        
        # Compute contrastive energy loss
        energy_stack = torch.cat([energy_pos, energy_neg], dim=-1)
        target = torch.zeros(energy_pos.size(0), device=energy_stack.device).long()
        loss_energy = F.cross_entropy(-1 * energy_stack, target, reduction='none')
        
        return loss_energy
    
    def compute_refinement_loss(self, x, t, labels):
        """
        Compute MCMC refinement loss for energy landscape learning.
        
        Args:
            x: Clean samples [B, C, H, W]
            t: Timesteps [B]
            labels: Class labels [B]
            
        Returns:
            MCMC refinement loss tensor
        """
        # Add opt-step refinement loss that mimics inference process
        noise = torch.randn_like(x)
        x_noisy_for_opt = self.train_diffusion.q_sample(x_start=x, t=t, noise=noise)
        
        # Apply opt_step with always_accept and get energy differences
        original_always_accept = self.always_accept_opt_steps
        self.always_accept_opt_steps = True
        
        x_refined, energy_trajectory = self.opt_step(x_noisy_for_opt.detach(), t, labels, step=None, keep_grad=True, return_energy_trajectory=True)
        
        # Restore original setting
        self.always_accept_opt_steps = original_always_accept
        
        # Loss: optimize for steepest energy descent trajectory
        # Calculate energy drops between consecutive steps
        energy_drops = energy_trajectory[:-1] - energy_trajectory[1:]  # Positive = energy reduction
        total_descent = energy_drops.sum()  # Total energy reduction
        consistency = -torch.var(energy_drops)  # Penalize erratic trajectories
        
        # Raw loss (maximize descent, minimize variance)
        raw_opt_loss = -total_descent + 0.1 * consistency
        
        # Normalize to 0-1 range using sigmoid to match MSE loss scale
        opt_refinement_loss = torch.sigmoid(raw_opt_loss / energy_trajectory.size(0))  # Scale by num steps and sigmoid
        
        return opt_refinement_loss
    
    def forward(self, x, labels, return_loss_dict=False):
        """
        Training forward pass. Compatible with MAR training pipeline.
        
        Args:
            x: VAE encoded latent tensor [B, C, H, W]
            labels: Class labels [B]
            return_loss_dict: If True, return dict with separate loss components
        
        Returns:
            Diffusion loss (or dict with loss components if return_loss_dict=True)
        """
        # Sample random timesteps
        bsz = x.shape[0]
        t = torch.randint(0, self.train_diffusion.num_timesteps, (bsz,), device=x.device)
        
        # Compute diffusion loss (both for standard and energy depending on the self.dit.use_energy)
        loss_dict = self.train_diffusion.training_losses(
            model=self.dit,
            x_start=x,
            t=t,
            model_kwargs={"y": labels}
        )
        loss_mse = loss_dict["loss"]
        total_loss = loss_mse.clone()
        
        # Optional loss components  
        if self.supervise_energy_landscape:
            contrastive_loss = self.compute_contrastive_loss(x, t, labels)
            total_loss += self.contrasive_loss_scale * contrastive_loss.mean()
        
        if self.learnable_mcmc_step_size:
            refinement_loss = self.compute_refinement_loss(x, t, labels)
            total_loss += self.mcmc_refinement_loss_scale * refinement_loss
        
        if return_loss_dict:
            loss_dict = {
                'total_loss': total_loss.mean(),
                'mse_loss': loss_mse.mean()
            }
            if self.supervise_energy_landscape:
                loss_dict['energy_loss'] = contrastive_loss.mean()
            if self.learnable_mcmc_step_size:
                loss_dict['opt_refinement_loss'] = refinement_loss
            return loss_dict
        else:
            return total_loss.mean()

    
    def get_adaptive_steps(self, t, keep_grad=False, base_steps=None):
        """
        Calculate adaptive number of optimization steps based on timestep and context.
        
        Args:
            t: Timestep tensor [B]
            keep_grad: If True (training), use fewer steps for efficiency
            base_steps: Override base step count
            
        Returns:
            Number of optimization steps (int)
        """
        if base_steps is None:
            if keep_grad:  # Training time - keep efficient
                base_steps = 2
            else:  # Inference time - be more thorough
                base_steps = 5
        
        # Timestep adaptation: more steps needed for higher timesteps (more noise)
        t_normalized = t.float().mean() / self.train_diffusion.num_timesteps
        timestep_factor = 1.0 + 2.0 * t_normalized  # 1.0 to 3.0 range
        
        adaptive_steps = int(base_steps * timestep_factor)
        return max(1, min(adaptive_steps, 10))  # Clamp between 1 and 10

    def opt_step(self, x, t, labels, step=None, step_size=None, keep_grad=False, mode='descent', return_accept_info=False, return_energy_trajectory=False):
        """
        Optimization step for energy diffusion following IRED approach.
        Performs gradient descent or ascent on energy landscape.
        
        Args:
            x: Current sample [B, C, H, W]
            t: Timestep [B]
            labels: Class labels [B]
            step: Number of optimization steps
            step_size: Step size for gradient optimization (if None, uses learnable alpha)
            keep_grad: If True, preserve gradients for alpha learning (used during training)
            mode: 'descent' for gradient descent, 'ascent' for gradient ascent
        
        Returns:
            If log_energy_accept_rate and not keep_grad: (optimized_sample, accept_count, total_count)
            Otherwise: optimized_sample
        """
        if not self.use_energy:
            return x
        
        # Calculate adaptive steps if not provided
        if step is None:
            step = self.get_adaptive_steps(t, keep_grad)
            
        # Use learnable alpha parameter, clamped to prevent instability (following DEBT pattern)
        alpha = torch.clamp(self.alpha, min=1e-4) if step_size is None else step_size
        
        # Initialize tracking
        total_accept_count = 0
        total_step_count = 0
        energy_trajectory = []  # For return_energy_trajectory
            
        with torch.enable_grad():
            x_opt = x.clone().requires_grad_(True)
            
            for i in range(step):
                # Get energy and gradients
                energy, gradients = self.dit(x_opt, t, labels, return_both=True)
                
                # Collect energy for trajectory tracking
                if return_energy_trajectory:
                    energy_trajectory.append(energy)
                #TODO: Greedy refinement?
                if mode == 'ascent':
                    x_new = x_opt + alpha * gradients.float()  # Gradient ascent for training
                else:
                    x_new = x_opt - alpha * gradients.float()  # Gradient descent for inference
                
                alpha_cumprod = torch.from_numpy(self.train_diffusion.alphas_cumprod).float().to(t.device)[t]
                max_val = torch.sqrt(alpha_cumprod).view(-1, 1, 1, 1) * 2.0
                x_new = torch.clamp(x_new, -max_val, max_val)
                
                # Check if energy decreased, if not, reject the step (unless always_accept_opt_steps is True)
                if not (self.use_innerloop_opt and self.always_accept_opt_steps):
                    energy_new = self.dit(x_new, t, labels, return_energy=True)
                    bad_step = (energy_new > energy).squeeze()
                    
                    # Track accept rate if enabled and not during training
                    if self.log_energy_accept_rate and not keep_grad:
                        accept_count = (~bad_step).sum().item()
                        total_count = bad_step.numel()
                        total_accept_count += accept_count
                        total_step_count += total_count
                    
                    # Keep old values where energy increased
                    if bad_step.any():
                        x_new[bad_step] = x_opt[bad_step]
                
                if keep_grad:
                    x_opt = x_new
                else:
                    x_opt = x_new.detach().requires_grad_(True)
        
        # Keep gradients for alpha learning during training, detach during sampling
        result = x_opt if keep_grad else x_opt.detach()
        
        # Return requested additional data
        if return_energy_trajectory:
            energy_trajectory = torch.stack(energy_trajectory, dim=0)  # Stack to tensor
            return result, energy_trajectory
        elif return_accept_info:
            return result, total_accept_count, total_step_count
        elif self.log_energy_accept_rate and not keep_grad:
            return result, total_accept_count, total_step_count
        else:
            return result
    
    def energy_aware_p_sample_loop(self, shape, labels, progress=False):
        """
        Modified p_sample_loop that includes energy optimization steps.
        Based on IRED's approach of calling opt_step during sampling.
        """
        device = next(self.dit.parameters()).device
        bsz = labels.shape[0]
        
        # Initialize with noise
        x = torch.randn(bsz, *shape, device=device)
        
        # Initialize accept rate tracking
        total_accept_count = 0
        total_step_count = 0
        
        # Use gen_diffusion for energy-aware sampling too
        timesteps = list(range(self.gen_diffusion.num_timesteps))[::-1]
        
        from tqdm import tqdm
        for i, t_val in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t = torch.full((bsz,), t_val, device=device, dtype=torch.long)
            
            # Standard diffusion sampling step using gen_diffusion
            with torch.no_grad():
                out = self.gen_diffusion.p_sample(
                    self.dit,
                    x,
                    t, 
                    model_kwargs={"y": labels}
                )
                x = out["sample"]
            
            if self.use_energy and self.use_innerloop_opt:
                # Use adaptive steps - automatically adjusts based on timestep and inference context
                if self.log_energy_accept_rate:
                    x, accept_count, step_count = self.opt_step(x, t, labels, step=None)
                    total_accept_count += accept_count
                    total_step_count += step_count
                else:
                    x = self.opt_step(x, t, labels, step=None)
        
        # Log overall accept rates after sampling completes
        if self.log_energy_accept_rate and total_step_count > 0:
            accept_rate_percent = (total_accept_count / total_step_count) * 100
            avg_accepted_per_pic = total_accept_count // bsz if bsz > 0 else 0
            total_steps_per_pic = total_step_count // bsz if bsz > 0 else 0
            print(f"Energy diffusion accept rate: {avg_accepted_per_pic}/{total_steps_per_pic} steps ({accept_rate_percent:.1f}%) across {bsz} pictures")
            
            # Log to wandb if available and run_name is set
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "energy_accept_rate": accept_rate_percent,
                        # "energy_accepted_steps": avg_accepted_per_pic,
                        # "energy_total_steps": total_steps_per_pic
                    })
            except (ImportError, AttributeError):
                pass  # wandb not available or not initialized
                
        return x
    
    def sample_tokens(
        self, 
        bsz, 
        num_iter=None,  # Ignored
        cfg=1.0, 
        cfg_schedule="linear",  # Ignored
        labels=None, 
        temperature=1.0,  # Ignored
        progress=False,
        gt_prefix_tokens=None,  # Ignored
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
            
            samples = self.gen_diffusion.p_sample_loop(
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
                # Use gen_diffusion with reduced timesteps
                samples = self.gen_diffusion.p_sample_loop(
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
    return PureDiffusion(dit_model="DiT-B/1", **kwargs)


def pure_diffusion_large(**kwargs):
    """Pure diffusion model with DiT-L backbone"""
    return PureDiffusion(dit_model="DiT-L/1", **kwargs)


def pure_diffusion_xlarge(**kwargs):
    """Pure diffusion model with DiT-XL backbone"""
    return PureDiffusion(dit_model="DiT-XL/1", **kwargs)
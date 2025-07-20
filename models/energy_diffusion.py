import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class EnergyDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        vae_stride,
        patch_size,
        timesteps=1000,
        sampling_timesteps=None,
        beta_schedule='cosine',
        use_innerloop_opt=False,
        ddim_sampling_eta=0.0,
        loss_type='l2',
        energy_loss_weight=1.0,
    ):
        super().__init__()
        self.model = model
        # Derive latent channels from DiT token dimension and patch_size
        # model.in_dim = channels * patch_size^2
        self.channels = model.in_dim // (patch_size * patch_size)
        self.image_size = image_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.latent_h = self.image_size // self.vae_stride
        self.latent_w = self.image_size // self.vae_stride
        self.latent_shape = (self.channels, self.latent_h, self.latent_w)
        self.seq_len = self.latent_h * self.latent_w
        self.in_dim = self.channels * self.patch_size * self.patch_size
        
        self.use_innerloop_opt = use_innerloop_opt
        self.energy_loss_weight = energy_loss_weight

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        register_buffer('opt_step_size', betas * torch.sqrt(1 / (1 - alphas_cumprod)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def patchify(self, x):
        """
        x: (N, C, H, W)
        imgs: (N, L, D)
        """
        c = self.channels
        p = self.patch_size
        h, w = self.latent_h, self.latent_w
        assert h * w == self.seq_len

        x = x.reshape(shape=(x.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(x.shape[0], h * w, c * p * p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, D)
        imgs: (N, C, H, W)
        """
        c = self.channels
        p = self.patch_size
        h, w = self.latent_h, self.latent_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, labels, clip_denoised=True):
        was_training = self.model.training
        self.model.eval()
        
        with torch.set_grad_enabled(True):
            _, _, pred_noise = self.model(x, t, labels)  # pred_noise: (B, S, D)
        
        if was_training:
            self.model.train()
            
        x_start = self.predict_start_from_noise(x, t, pred_noise) # x_start: (B, S, D)

        if clip_denoised:
            x_start.clamp_(-3., 3.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)) # noise: (B, S, D)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        ) # (B, S, D)

    def p_losses(self, x_start, t, *, labels):
        x_start_tokens = self.patchify(x_start) # (B, S, D)
        b, l, d = x_start_tokens.shape
        noise = torch.randn_like(x_start_tokens) # (B, S, D)

        x_t = self.q_sample(x_start=x_start_tokens, t=t, noise=noise) # (B, S, D)

        energy, opt_grad, pred_noise = self.model(x_t, t, labels) # energy: (B, 1), opt_grad: (B, S, D), pred_noise: (B, S, D)

        if self.loss_type == 'l1':
            loss_denoise = (pred_noise - noise).abs().mean()
        elif self.loss_type == 'l2':
            loss_denoise = F.mse_loss(pred_noise, noise)
        else:
            raise NotImplementedError()

        if self.energy_loss_weight > 0:
            data_sample = x_t.clone().detach() # (B, S, D)

            neg_noise = torch.randn_like(x_t) # (B, S, D)
            xmin_noise = self.q_sample(x_start=x_start_tokens, t=t, noise=3.0 * neg_noise) # (B, S, D)

            if self.use_innerloop_opt:
                with torch.set_grad_enabled(True):
                    _, grad, _ = self.model(xmin_noise, t, labels)  # grad: (B, S, D)
                xmin_noise = xmin_noise - extract(self.opt_step_size, t, grad.shape) * grad  # (B, S, D)
            
            # energy loss
            energy_data = self.model(data_sample, t, labels)[0]  # (B, 1)
            energy_noise = self.model(xmin_noise.detach(), t, labels)[0]  # (B, 1)
            
            loss_energy = (energy_data - energy_noise).mean()
            
            return loss_denoise + self.energy_loss_weight * loss_energy
        else:
            return loss_denoise

    def opt_step(self, img_tokens, t, labels, step=5, sf=1.0):
        for i in range(step):
            img_tokens = img_tokens.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                energy, grad, _ = self.model(img_tokens, t, labels) # energy: (B, 1), grad: (B, S, D)
            img_new_tokens = img_tokens - extract(self.opt_step_size, t, grad.shape) * grad * sf # (B, S, D)
            
            # This clamp is on token space, may need adjustment.
            max_val = extract(self.sqrt_recip_alphas_cumprod, t, img_new_tokens.shape)
            img_new_tokens = torch.clamp(img_new_tokens, -max_val, max_val)
            
            energy_new, _, _ = self.model(img_new_tokens, t, labels) # energy_new: (B, 1)
            
            bad_step = (energy_new.squeeze() > energy.squeeze())  # (B,)
            
            img_new_tokens[bad_step] = img_tokens[bad_step]
            img_tokens = img_new_tokens.detach()
        return img_tokens
    
    @torch.no_grad()
    def p_sample(self, x, t, labels, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        x_tokens = self.patchify(x)
        
        if isinstance(t, int):
            batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        else:
            batched_times = t
            
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x_tokens, t=batched_times, labels=labels, clip_denoised=clip_denoised)
        
        pred_img_tokens = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start
        
        pred_img = self.unpatchify(pred_img_tokens)
        x_start_img = self.unpatchify(x_start)  # Convert x_start back to image format
        return pred_img, x_start_img

    @torch.no_grad()
    def p_sample_loop(self, shape, labels, num_sampling_steps=250):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device) # img: (B, C, H, W)

        # Create sampling schedule: from T-1 down to 0
        if num_sampling_steps >= self.num_timesteps:
            # Full sampling schedule
            timesteps = list(reversed(range(0, self.num_timesteps)))
        else:
            # Subsampled schedule: select num_sampling_steps evenly spaced timesteps
            timesteps = torch.linspace(0, self.num_timesteps - 1, num_sampling_steps).long()
            timesteps = list(reversed(timesteps.tolist()))
        
        iterator = tqdm(timesteps, desc='sampling loop time step', total=len(timesteps))

        for t_int in iterator:
            t = torch.full((b,), t_int, device=device, dtype=torch.long) # t: (B,)

            img, x_start_img = self.p_sample(img, t_int, labels=labels)
            
            if self.use_innerloop_opt and t_int > 0: # no opt at last step
                img_tokens = self.patchify(img)
                img_tokens = self.opt_step(img_tokens, t, labels, step=5) # img_tokens: (B, S, D)
                img = self.unpatchify(img_tokens)
            
            if t_int != 0:
                img_tokens = self.patchify(img)
                
                img_tokens_unscaled = self.predict_start_from_noise(img_tokens, t, torch.zeros_like(img_tokens))
                
                
                img_tokens_unscaled = torch.clamp(img_tokens_unscaled, -4.0, 4.0)
                
                batched_times_prev = t - 1
                img_tokens_scaled = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_tokens_unscaled.shape) * img_tokens_unscaled
                
                img = self.unpatchify(img_tokens_scaled)
        
        return img

    def forward(self, x, labels, *args, **kwargs):
        b, c, h, w = x.shape # x: (B, C, H, W)
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() # t: (B,)
        return self.p_losses(x, t, labels=labels, *args, **kwargs)

    @torch.no_grad()
    def sample_tokens(self, bsz, labels, num_iter=None, temperature=1.0, progress=True, **kwargs):
        """
        The `sample_tokens` name is kept for compatibility with `engine_mar.py`.
        It doesn't sample tokens autoregressively but performs diffusion sampling.
        `num_iter` corresponds to the number of sampling timesteps.
        """
        sampling_timesteps = default(num_iter, self.sampling_timesteps)
        
        shape = (bsz, self.channels, self.latent_h, self.latent_w)
        sampled_latents = self.p_sample_loop(shape, labels, num_sampling_steps=sampling_timesteps) # (B, C, H, W)
        return sampled_latents 
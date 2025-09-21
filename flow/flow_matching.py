import torch
from flow.scheduler import NoiseScheduler
import torch.nn.functional as F
from flow_matching.solver.ode_solver import ODESolver

class FlowMatching:
    def __init__(
            self, noise_scheduler: NoiseScheduler, timesteps: int,
            ode_method, ode_step_size
    ):
        self.noise_scheduler = noise_scheduler
        self.norm_timestep = timesteps - 1
        self.ode_method = ode_method
        self.ode_step_size = ode_step_size

    def norm_reverse(self, timestep: int):
        return 1.0 - timestep / self.norm_timestep

    def generate_noisy_samples(self, x_start, t, noise):
        x_t, u_t = self.noise_scheduler.sample(
                t=t,
                samples=-x_start,
                noise=noise,
        )

        return x_t, u_t

    def training_loss(self, model, x_start, t, model_kwargs=None):
            norm_t = self.norm_reverse(t)
            noise = torch.randn_like(x_start, device=x_start.device)
            x_t, u_t = self.generate_noisy_samples(x_start, norm_t, noise)
            labels = model_kwargs.get('y', None)
            out = model(x_t, norm_t, labels)
            diff = out - u_t
            per_sample = diff.reshape(diff.shape[0], -1).pow(2).mean(dim=1)
            loss = per_sample.mean()
            return {'loss': loss}
    
    def solve(self, x, t, labels, **kwargs):
        device = x.device
        norm_t = self.norm_reverse(t)
        
        model = kwargs['model']
        
        vf = self.noise_scheduler.get_velocity_function(model)
        solver = ODESolver(velocity_model=vf)
        time_grid = torch.tensor([norm_t, 1.0], device=device)

        synthetic_samples = solver.sample(
            time_grid=time_grid,
            x_init=x,
            method=self.ode_method,
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.ode_step_size,
            y=labels,
        )

        return synthetic_samples
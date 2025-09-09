from .flow_matching import FlowMatching
from .scheduler import NoiseScheduler


def create_flow(
        timesteps,
        ode_method,
        ode_step_size
):
    obj = FlowMatching(
        NoiseScheduler(), timesteps, 
        ode_method, ode_step_size
    )

    return obj

from gymnasium.envs.registration import register
from . import envs

register(
    id="airsim-drone-sample-v0",
    entry_point="sim.reinforcement_learning.airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-car-sample-v0",
    entry_point="sim.reinforcement_learning.airgym.envs:AirSimCarEnv",
)

register(
    id="airsim-drone-ppo-v0",
    entry_point="sim.reinforcement_learning.airgym.envs:AirSimDronePPOEnv",
)

__all__ = ['envs']
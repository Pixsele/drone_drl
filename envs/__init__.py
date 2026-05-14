from gymnasium.envs.registration import register
from .drone_env import DroneDirectionBaseEnv
from .drone_qr_env import DroneDirectionQREnv

register(
    id="drone-env-direction-base",
    entry_point="envs:DroneDirectionBaseEnv",
)

register(
    id="drone-env-qr",
    entry_point="envs:DroneDirectionQREnv",
)

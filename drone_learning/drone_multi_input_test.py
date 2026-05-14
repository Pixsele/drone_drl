import time

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv

from drone_learning.train.augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper, \
    GaussianNoiseWrapper, DepthQuantizationWrapper, DistortionWrapper, RgbToDepthWrapper
from sim.reinforcement_learning.airgym.envs import AirSimDronePPOEnv, AirSimDroneDirectionPPOEnv
from other import DroneExtractor
angle =  3 * np.pi / 2

reward_params = {
    "direction": [float(np.cos(angle)), float(np.sin(angle)), 0.0],  # направление

    "progress_reward": 1.0,
    "target_reward": 10.0,

    "lateral_penalty": 0.8,
    "collision_fine": 5.0,  # штраф за столкновение

    "vx": 2.5,  #
    "vy": 2.5,  #
    "vz": 1.0,  #

    "max_steps": 300,  # максимум шагов за эпизод
    "target_distance": 100.0,
    "stall_steps": 20,  # шагов на зависание для завершения эпизода
    "stall_fine": 3.0,  # штраф за полное зависание

    "direction_rand_frequency": 20,
    "altitude_penalty": 0.5
}

env = AirSimDroneDirectionPPOEnv(
    ip_address="127.0.0.1",
    step_length=0.2,
    image_shape=(128, 128, 3),
    params=reward_params,
    run_name=None,
    eval=False,
)

# env = RandomShiftWrapper(env,10)
env = RgbToDepthWrapper(env)
# env = SaltPepperWrapper(env, 0.01)
# env = CutWrapper(env,10,3)
# env = GaussianNoiseWrapper(env, 0.01)
# env = DepthQuantizationWrapper(env,128)
# env = DistortionWrapper(env,0.2)


vec_env = DummyVecEnv([lambda: env])
# vec_env = VecTransposeImage(vec_env)

model = PPO.load(
    "train/models/PPO_DirectionV3_01_05_2026__18-06-36/best_model"
)

obs = vec_env.reset()

new_obs = {
    "image": obs["image"],
    "direction": reward_params["direction"],
}



print("obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))
print("direction в obs:", new_obs["direction"] if isinstance(obs, dict) else "нет direction!")
i = 0

while True:
    action, _ = model.predict(new_obs, deterministic=True)

    new_obs, reward, done, info = vec_env.step(action)

    depth_img = new_obs["image"].squeeze()
    # cv2.imwrite(f"look/{i}.jpg", depth_img)

    i += 1
    print("reward:", reward[0])

    if done:
        print("Episode finished")
        obs= vec_env.reset()
        time.sleep(2)
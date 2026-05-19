import time

import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from drone_learning.wrappers.augmentation_obs import DistortionWrapper, RgbToDepthWrapper
from sim.reinforcement_learning.airgym.envs import AirSimDronePPOEnv


reward_params = {
    "target": (100.0, 0.0, -5.0),       # цель
    "dist_reward": -0.01,               # коэф. штраф за дистанцию за шаг
    "dir_to_target_reward": 0.3,        # коэф. награда за движение в направлении к цели
    "lateral_speed_reward": 0.015,      # коэф. штрафа за боковую скорость
    "diff_dist_reward": 1.5,            # коэф. награды за движение
    "diff_dist_not_fine": 0.02,         # штраф за зависание
    "collision_fine": 20.0,             # штраф за столкновение
    "target_reward": 30.0,              # награда за достижение цели
    "vx": 2.5,                          #
    "vy": 2.5,                          #
    "vz": 1.0,                          #
    "max_steps": 500,                   # максимум шагов за эпизод
    "stall_speed": 0.05,                # ниже этого считается зависанием
    "stall_steps": 30,                  # шагов на зависание для завершения эпизода
    "max_step_fine": 5.0,               # штраф за большое кол-во шагов
    "stall_fine": 10.0                  # штраф за полное зависание
}

env = AirSimDronePPOEnv(
    ip_address="127.0.0.1",
    step_length=0.2,
    image_shape=(128, 128, 3),
    params=reward_params,
)

# env = RandomShiftWrapper(env,10)

env = RgbToDepthWrapper(env)
# env = SaltPepperWrapper(env, 0.01)
# env = CutWrapper(env,10,3)
# env = GaussianNoiseWrapper(env, 0.01)
# env = DepthQuantizationWrapper(env,128)
env = DistortionWrapper(env,0.2)

vec_env = DummyVecEnv([lambda: env])
vec_env = VecTransposeImage(vec_env)

model = PPO.load("train/models/PPO_DepthAnythingTest_09_04_2026__17-23-07/best_model")

obs = vec_env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info= vec_env.step(action)

    depth_img = obs[0].squeeze()
    cv2.imwrite(f"images/test_mix_{time.time()}.jpg", depth_img)

    print("reward:", reward[0])

    if done:
        print("Episode finished")
        obs= vec_env.reset()
        time.sleep(2)
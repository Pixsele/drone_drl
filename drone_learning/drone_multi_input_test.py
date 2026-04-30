import time

import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv

from drone_learning.train.augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper, \
    GaussianNoiseWrapper, DepthQuantizationWrapper, DistortionWrapper, RgbToDepthWrapper
from sim.reinforcement_learning.airgym.envs import AirSimDronePPOEnv, AirSimDroneDirectionPPOEnv

reward_params = {
    "direction": [-1.0, -1.0 , 0.0],  # направление

    "alignment_reward": 0.3,
    "progress_reward": 2.0,
    "target_reward": 200.0,

    "lateral_penalty": 0.05,
    "collision_fine": 100.0,  # штраф за столкновение

    "vx": 2.5,  #
    "vy": 2.5,  #
    "vz": 1.0,  #

    "max_steps": 300,  # максимум шагов за эпизод
    "target_distance": 100.0,
    "stall_steps": 20,  # шагов на зависание для завершения эпизода
    "stall_fine": 10.0,  # штраф за полное зависание

    "direction_rand_frequency": 50,
    "altitude_penalty": 1.0
}

env = AirSimDroneDirectionPPOEnv(
    ip_address="127.0.0.1",
    step_length=0.2,
    image_shape=(128, 128, 1),
    params=reward_params,
    run_name=None,
)

# env = RandomShiftWrapper(env,10)
# env = RgbToDepthWrapper(env)
# env = SaltPepperWrapper(env, 0.01)
# env = CutWrapper(env,10,3)
# env = GaussianNoiseWrapper(env, 0.01)
# env = DepthQuantizationWrapper(env,128)
# env = DistortionWrapper(env,0.2)


vec_env = DummyVecEnv([lambda: env])
# vec_env = VecTransposeImage(vec_env)

model = PPO.load("train/models/Test_PPO_DirectionV1_30_04_2026__00-46-49/best_model")

obs = vec_env.reset()
print("obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))
print("direction в obs:", obs["direction"] if isinstance(obs, dict) else "нет direction!")


while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = vec_env.step(action)

    # depth_img = obs[0].squeeze()
    # cv2.imwrite(f"images/test_mix_{time.time()}.jpg", depth_img)

    print("reward:", reward[0])

    if done:
        print("Episode finished")
        obs= vec_env.reset()
        time.sleep(2)
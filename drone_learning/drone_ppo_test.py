import time

import cv2
from stable_baselines3 import PPO

from drone_learning.train.augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper
from sim.reinforcement_learning.airgym.envs import AirSimDronePPOEnv

env = AirSimDronePPOEnv(
    ip_address="127.0.0.1",
    step_length=0.2,
    image_shape=(128, 128, 1)
)

env = RandomShiftWrapper(env,6)
env = SaltPepperWrapper(env, 0.01)
env = CutWrapper(env,10,3)

model = PPO.load("models/PPO_clear_20_03_2026__20-07-39/best_model")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    cv2.imwrite(f"images/obs_shift_{time.time()}.jpg", obs)
    cv2.waitKey(0)

    print("reward:", reward)

    if terminated or truncated:
        print("Episode finished")
        obs, _ = env.reset()
        time.sleep(2)
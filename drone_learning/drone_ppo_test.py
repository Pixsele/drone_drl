import time

import cv2
from stable_baselines3 import PPO

from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper
from sim.reinforcement_learning.airgym.envs import AirSimDronePPOEnv

env = AirSimDronePPOEnv(
    ip_address="127.0.0.1",
    step_length=0.2,
    image_shape=(84, 84, 1)
)

env = RandomShiftWrapper(env)
env = SaltPepperWrapper(env, 0.01)

model = PPO.load("best_model/best_model_ppo.zip")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    cv2.imwrite(f"images/obs_{time.time()}.jpg", obs)
    cv2.waitKey(0)

    print("reward:", reward)

    if terminated or truncated:
        print("Episode finished")
        obs, _ = env.reset()
        time.sleep(2)
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

import sim.reinforcement_learning.airgym
from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper


def make_env():
    env_new = gym.make("airsim-drone-ppo-v0",
                   ip_address="127.0.0.1",
                   step_length=0.25,
                   image_shape=(84,84,1))
    env_new = RandomShiftWrapper(env_new)
    env_new = SaltPepperWrapper(env_new, 0.01)
    return env_new

if __name__ == "__main__":
    check_env(make_env(), warn=True)

    # env = SubprocVecEnv([lambda: make_env() for _ in range(1)])
    env = DummyVecEnv([lambda: make_env()])

    env = VecTransposeImage(env)

    eval_env = VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env())]))

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log="./tb_logs/",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=2048
    )

    run_name = f"PPO_{datetime.now().strftime('%Y_%m_%d__%H-%M-%S')}"
    new_logger = configure(f"./tb_logs/{run_name}", ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    params = env.envs[0].unwrapped.params_reward
    for k, v in params.items():
        model.logger.record(f"env/{k}", v)
    model.logger.dump(step=0)

    model.learn(total_timesteps=50_000, callback=eval_callback)

    model.save("ppo_airsim_drone_policy")
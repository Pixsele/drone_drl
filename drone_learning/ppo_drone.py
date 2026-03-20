import glob
import os
import time
import shutil
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import sim.reinforcement_learning.airgym
from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "success" not in info:
                continue

            self.episode_metrics.append({
                "success":          float(info.get("success", False)),
                "collision":        float(info.get("collision", False)),
                "stall":            float(info.get("stall", False)),
                "steps_to_goal":    info.get("steps_to_goal", None),
                "path_efficiency":  info.get("path_efficiency", None),
            })

            window = self.episode_metrics[-20:]

            self.logger.record("metrics/success_rate",
                sum(m["success"] for m in window) / len(window))
            self.logger.record("metrics/collision_rate",
                sum(m["collision"] for m in window) / len(window))
            self.logger.record("metrics/stall_rate",
                sum(m["stall"] for m in window) / len(window))

            steps = [m["steps_to_goal"] for m in window if m["steps_to_goal"]]
            if steps:
                self.logger.record("metrics/steps_to_goal", np.mean(steps))

            eff = [m["path_efficiency"] for m in window if m["path_efficiency"]]
            if eff:
                self.logger.record("metrics/path_efficiency", np.mean(eff))

        return True

ppo_params = {
    "total_timesteps": 100_000,
    "step_length": 0.25,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "n_steps": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "image_shape": (128, 128, 1)
}

def to_hparam(v):
    if isinstance(v, tuple):
        return str(v)
    if isinstance(v, bool):
        return v
    if hasattr(v, '__float__'):
        return float(v)
    return str(v)

def make_env():
    env_new = gym.make("airsim-drone-ppo-v0",
                   ip_address="127.0.0.1",
                   step_length=ppo_params["step_length"],
                   image_shape=ppo_params["image_shape"],)
    # env_new = RandomShiftWrapper(env_new)
    # env_new = SaltPepperWrapper(env_new, 0.01)
    return env_new

if __name__ == "__main__":
    run_name = f"PPO_clear_{datetime.now().strftime('%d_%m_%Y__%H-%M-%S')}"
    log_dir = f"./tb_logs/{run_name}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    os.makedirs(f"./models/{run_name}", exist_ok=True)

    check_env(make_env(), warn=True)

    # env = SubprocVecEnv([lambda: make_env() for _ in range(1)])
    env = DummyVecEnv([lambda: make_env()])

    env = VecTransposeImage(env)

    eval_env = VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env())]))

    env_params = env.envs[0].unwrapped.params


    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=ppo_params["learning_rate"],
        batch_size=ppo_params["batch_size"],
        n_steps=ppo_params["n_steps"],
        n_epochs=ppo_params["n_epochs"],
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params["gae_lambda"],
        clip_range=ppo_params["clip_range"],
        ent_coef=ppo_params["ent_coef"],
        vf_coef=ppo_params["vf_coef"],
        max_grad_norm=ppo_params["max_grad_norm"],
        device="cuda",
        tensorboard_log="./tb_logs/",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        best_model_save_path=f"./models/{run_name}",
        log_path=f"./models/{run_name}",
        eval_freq=ppo_params["n_steps"],
    )

    metrics_callback = MetricsCallback()

    model.set_logger(new_logger)

    try:
        model.learn(total_timesteps=ppo_params["total_timesteps"], callback=[eval_callback,metrics_callback])
        model.save(f"./models/{run_name}/model_final")
    finally:
        all_params = {**env_params, **ppo_params}
        all_params_clear = {k: to_hparam(v) for k, v in all_params.items()}
        writer = SummaryWriter(log_dir, filename_suffix=".hparams")
        writer.add_hparams(hparam_dict=all_params_clear, metric_dict={})
        writer.close()

        for subdir in glob.glob(f"{log_dir}/*/"):
            for f in glob.glob(f"{subdir}events.out.tfevents.*"):
                shutil.move(f, log_dir)
            shutil.rmtree(subdir)

        model.save(f"./models/{run_name}/model_interrupted")


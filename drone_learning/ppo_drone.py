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
from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper
from drone_learning.log_helpers import MetricsCallback, to_hparam

n_env = 2

ppo_params = {
    "total_timesteps": 100_000,         # кол-во шагов
    "step_length": 0.25,                # длина шага агента
    "learning_rate": 3e-4,              # скорость обучения
    "batch_size": 256,                  # размер батча
    "n_steps": 2048,                    # шагов на обновление
    "n_epochs": 10,                     # эпох на обновление
    "gamma": 0.99,                      # коэф. дисконтирования
    "gae_lambda": 0.95,                 # lambda
    "clip_range": 0.2,                  # клиппинг PPO
    "ent_coef": 0.01,                   # коэф. энтропии
    "vf_coef": 0.5,                     # коэф. функции ценности
    "max_grad_norm": 0.5,               # макс. норма градиента
    "image_shape": (128, 128, 1)        # размер входного изображения
}

wrapper_params = {
    "pad_shift": 6,                     # размер отступа
    "salt_pepper": 0.01,                # коэф. закрашенных пикселей
    "cut_size": 15,                     # макс. размер вырезки
    "cut_count": 3                      # кол-во вырезок
}

def make_env(drone_id):
    env_new = gym.make("airsim-drone-ppo-v0",
                       ip_address="127.0.0.1",
                       step_length=ppo_params["step_length"],
                       image_shape=ppo_params["image_shape"],
                       drone_id=drone_id,)

    env_new = RandomShiftWrapper(env_new, wrapper_params["pad_shift"])
    env_new = SaltPepperWrapper(env_new, wrapper_params["salt_pepper"])
    env_new = CutWrapper(env_new, wrapper_params["cut_size"], wrapper_params["cut_count"])

    return env_new

if __name__ == "__main__":
    run_name = f"PPO_test_{datetime.now().strftime('%d_%m_%Y__%H-%M-%S')}"
    log_dir = f"./tb_logs/{run_name}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    os.makedirs(f"./models/{run_name}", exist_ok=True)

    # env = SubprocVecEnv([lambda: make_env() for _ in range(1)])
    # env = DummyVecEnv([
    #     lambda i=i: make_env(i) for i in range(n_env)
    # ])

    env = SubprocVecEnv([
        lambda i=i: make_env(i) for i in range(n_env)
    ])

    env = VecTransposeImage(env)

    eval_env = VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env(n_env))]))

    # TODO
    env_params = {}


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
        all_params = {**env_params, **ppo_params, **wrapper_params}
        all_params_clear = {k: to_hparam(v) for k, v in all_params.items()}
        writer = SummaryWriter(log_dir, filename_suffix=".hparams")
        writer.add_hparams(hparam_dict=all_params_clear, metric_dict={})
        writer.close()

        for subdir in glob.glob(f"{log_dir}/*/"):
            for f in glob.glob(f"{subdir}events.out.tfevents.*"):
                shutil.move(f, log_dir)
            shutil.rmtree(subdir)

        model.save(f"./models/{run_name}/model_interrupted")

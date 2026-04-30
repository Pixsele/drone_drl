import glob
import json
import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper, RgbToDepthWrapper, GaussianNoiseWrapper, \
    DepthQuantizationWrapper, DistortionWrapper
from drone_learning.train.log_helpers import MetricsCallback, to_hparam
from sim.reinforcement_learning.airgym.envs import AirSimDroneDirectionPPOEnv

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
    "image_shape": (128, 128, 1),       # размер входного изображения
    "n_env": 0,                        # кол-во дронов
}

wrapper_params = {
    "pad_shift": 6,                     # размер отступа
    "salt_pepper": 0.01,                # коэф. закрашенных пикселей
    "cut_size": 15,                     # макс. размер вырезки
    "cut_count": 3,                      # кол-во вырезок
    "quantize_level": 32,
    "gaussian": 0.05,
    "distortion": 0.5,
}

reward_params = {
    "direction": [1.0, 0.0, 0.0],  # направление

    "alignment_reward": 1.0,
    "progress_reward": 2.0,
    "target_reward": 200.0,

    "lateral_penalty": 0.5,
    "collision_fine": 100.0,  # штраф за столкновение
    "wrong_direction_penalty": 1.0,

    "vx": 2.5,  #
    "vy": 2.5,  #
    "vz": 1.0,  #

    "max_steps": 300,  # максимум шагов за эпизод
    "target_distance": 100.0,
    "stall_steps": 20,  # шагов на зависание для завершения эпизода
    "stall_fine": 10.0,  # штраф за полное зависание

    "direction_rand_frequency": 10,
    "altitude_penalty": 1.0
}

def make_env(drone_id, run_name):
    env_new = gym.make("airsim-drone-direction-ppo-v0",
                       ip_address="127.0.0.1",
                       step_length=ppo_params["step_length"],
                       image_shape=ppo_params["image_shape"],
                       params=reward_params,
                       run_name=run_name,
                       client_id=drone_id
                       )

    # env_new = RandomShiftWrapper(env_new, wrapper_params["pad_shift"])
    # env_new = SaltPepperWrapper(env_new, wrapper_params["salt_pepper"])
    # env_new = CutWrapper(env_new, wrapper_params["cut_size"], wrapper_params["cut_count"])
    # env_new = RgbToDepthWrapper(env_new)

    # env_new = GaussianNoiseWrapper(env_new, wrapper_params["gaussian"]) #первый
    # env_new = DepthQuantizationWrapper(env_new,wrapper_params["quantize_level"]) #второй
    # env_new = DistortionWrapper(env_new,wrapper_params["distortion"]) #третий
    return env_new

if __name__ == "__main__":
    run_name = f"Test_PPO_DirectionV2_{datetime.now().strftime('%d_%m_%Y__%H-%M-%S')}"
    log_dir = f"./tb_logs/{run_name}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    os.makedirs(f"./models/{run_name}", exist_ok=True)
    os.makedirs(f"./models/{run_name}/images", exist_ok=True)
    metrics_callback = MetricsCallback()

    # env = VecTransposeImage(
    #     SubprocVecEnv(
    #         [lambda i=i: make_env(i) for i in range(1, ppo_params["n_env"] + 1)]
    #     )
    # )

    # env = VecTransposeImage(
    #     SubprocVecEnv(
    #         [lambda: make_env(0)]
    #     )
    # )

    env = SubprocVecEnv(
        [lambda: make_env(0, run_name)],
    )

    eval_env = VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env(0, run_name))]))

    model = PPO(
        "MultiInputPolicy",
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

    model.set_logger(new_logger)

    try:
        model.learn(total_timesteps=ppo_params["total_timesteps"], callback=[eval_callback,metrics_callback])
        model.save(f"./models/{run_name}/model_final")
    finally:
        all_params = {**ppo_params, **ppo_params, **wrapper_params}
        all_params_clear = {k: to_hparam(v) for k, v in all_params.items()}
        writer = SummaryWriter(log_dir, filename_suffix=".hparams")
        writer.add_hparams(hparam_dict=all_params_clear, metric_dict={})
        writer.close()

        with open(f"./models/{run_name}/params.json", "w") as fp:
            json.dump(all_params_clear, fp)

        for subdir in glob.glob(f"{log_dir}/*/"):
            for f in glob.glob(f"{subdir}events.out.tfevents.*"):
                shutil.move(f, log_dir)
            shutil.rmtree(subdir)

        model.save(f"./models/{run_name}/model_interrupted")

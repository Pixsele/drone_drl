import glob
import json
import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from augmentation_obs import RandomShiftWrapper, SaltPepperWrapper, CutWrapper, RgbToDepthWrapper, GaussianNoiseWrapper, \
    DepthQuantizationWrapper, DistortionWrapper
from drone_learning.extractors.qr_extractor import QRExtractor
from drone_learning.other import DroneExtractor
from drone_learning.train.log_helpers import MetricsCallback, to_hparam
from sim.reinforcement_learning.airgym.envs import AirSimDroneDirectionPPOEnv
from envs.drone_env import DroneDirectionBaseEnv

# ppo_params = {
#     "total_timesteps": 100_000,         # кол-во шагов
#     "step_length": 0.25,                # длина шага агента
#     "learning_rate": 1e-4,              # скорость обучения
#     "batch_size": 256,                  # размер батча
#     "n_steps": 2048,                    # шагов на обновление
#     "n_epochs": 10,                     # эпох на обновление
#     "gamma": 0.99,                      # коэф. дисконтирования
#     "gae_lambda": 0.95,                 # lambda
#     "clip_range": 0.1,                  # клиппинг PPO
#     "ent_coef": 0.01,                   # коэф. энтропии
#     "vf_coef": 0.5,                     # коэф. функции ценности
#     "max_grad_norm": 0.5,               # макс. норма градиента
#     "image_shape": (128, 128, 3),       # размер входного изображения
#     "n_env": 0,                        # кол-во дронов
# }

wrapper_params = {
    "pad_shift": 6,                     # размер отступа
    "salt_pepper": 0.01,                # коэф. закрашенных пикселей
    "cut_size": 15,                     # макс. размер вырезки
    "cut_count": 3,                      # кол-во вырезок
    "quantize_level": 32,
    "gaussian": 0.05,
    "distortion": 0.5,
}
# Для простой direction
# reward_params = {
#     "direction": [1.0, 0.0, 0.0],  # направление
#
#     "progress_reward": 1.0,
#     "target_reward": 10.0,
#
#     "lateral_penalty": 0.8,
#     "collision_fine": 6.0,  # штраф за столкновение
#
#     "vx": 2.5,  #
#     "vy": 2.5,  #
#     "vz": 1.0,  #
#
#     "max_steps": 200,  # максимум шагов за эпизод
#     "target_distance": 100.0,
#     "stall_steps": 20,  # шагов на зависание для завершения эпизода
#     "stall_fine": 3.0,  # штраф за полное зависание
#
#     "direction_rand_frequency": 20,
#     "altitude_penalty": 0.65
# }

# Для QR
reward_params = {
    "direction": [1.0, 0.0, 0.0],  # направление

    "progress_reward": 5.0,

    "visible_reward": 0.5,
    "invisible_penalty": 1.0,

    "dist_penalty": 0.05,

    "collision_fine": 50.0,  # штраф за столкновение

    "land_threshold": 2.0,
    "land_reward": 100.0,

    "vx": 2.5,  #
    "vy": 2.5,  #
    "vz": 1.0,  #

    "max_steps": 70,  # максимум шагов за эпизод
}

ppo_params = {
    "total_timesteps": 100_000,         # кол-во шагов
    "step_length": 0.25,                # длина шага агента
    "learning_rate": 1e-4,              # скорость обучения
    "batch_size": 256,                  # размер батча
    "n_steps": 2048,                    # шагов на обновление
    "n_epochs": 10,                     # эпох на обновление
    "gamma": 0.99,                      # коэф. дисконтирования
    "gae_lambda": 0.95,                 # lambda
    "clip_range": 0.1,                  # клиппинг PPO
    "ent_coef": 0.01,                   # коэф. энтропии
    "vf_coef": 0.5,                     # коэф. функции ценности
    "max_grad_norm": 0.5,               # макс. норма градиента
    "image_shape": (256, 256, 3),       # размер входного изображения
    "n_env": 0,                        # кол-во дронов
}


# def make_env(drone_id, run_name, eval):
#     env_new = gym.make("drone-env-v0",
#                        ip_address="127.0.0.1",
#                        step_length=ppo_params["step_length"],
#                        image_shape=ppo_params["image_shape"],
#                        params=reward_params,
#                        run_name=run_name,
#                        client_id=drone_id,
#                        eval=eval,
#                        )

def make_env(drone_id, run_name, eval):
    env_new = gym.make("drone-env-qr",
                       ip_address="127.0.0.1",
                       image_shape=ppo_params["image_shape"],
                       params=reward_params,
                       run_name=run_name,
                       eval=eval,
                       camera_name="bottom_center",
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
    run_name = f"Test_{datetime.now().strftime('%d_%m_%Y__%H-%M-%S')}"
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

    env = VecTransposeImage(SubprocVecEnv([lambda: make_env(0, run_name, eval=False)]))
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = VecNormalize(
        VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env(0, run_name, eval=True))])),
        norm_obs=False,
        norm_reward=False,
        clip_reward=10.0
    )

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
        verbose=1,
        policy_kwargs={
            "features_extractor_class": QRExtractor,
            "features_extractor_kwargs": {
                "cnn_output_dim": 256,
                "direction_output_dim": 64
            }
        }
    )

    print(model.policy)

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
        all_params = {**ppo_params, **reward_params, **wrapper_params}
        all_params_clear = {k: to_hparam(v) for k, v in all_params.items()}
        writer = SummaryWriter(log_dir, filename_suffix=".hparams")
        writer.add_hparams(hparam_dict=all_params_clear, metric_dict={})
        writer.close()

        with open(f"./models/{run_name}/params.json", "w") as fp:
            json.dump(all_params_clear, fp, indent=4)

        for subdir in glob.glob(f"{log_dir}/*/"):
            for f in glob.glob(f"{subdir}events.out.tfevents.*"):
                shutil.move(f, log_dir)
            shutil.rmtree(subdir)

        model.save(f"./models/{run_name}/model_interrupted")

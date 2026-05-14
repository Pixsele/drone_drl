from datetime import datetime

from drone_learning.extractors.qr_extractor import QRExtractor
from drone_learning.train.augmentation_obs import RgbToDepthWrapper
from drone_learning.train.drone_train_base import DroneTrainBase

from envs.drone_qr_env import DroneDirectionQREnv

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
}

drone_params = {
    "camera_name": "bottom_center",
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

wrappers = [
    # (RgbToDepthWrapper, {}),
]

if __name__ == "__main__":
    run_name = f"QR_Land_v1_{datetime.now().strftime('%d_%m_%Y__%H-%M-%S')}"

    trainer = DroneTrainBase(
        run_name=run_name,
        ppo_params=ppo_params,
        drone_params=drone_params,
        wrappers=wrappers,
    )

    trainer.init_logger()

    trainer.make_vec_env(
        "drone-env-qr"
    )

    trainer.make_model(
        QRExtractor,
        {
            "cnn_output_dim": 512,
            "direction_output_dim": 64,
        }
    )

    trainer.train()



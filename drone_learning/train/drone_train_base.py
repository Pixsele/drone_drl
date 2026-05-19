import glob
import json
import os
import shutil

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, VecNormalize, DummyVecEnv

from drone_learning.logging_config import MetricsCallback, to_hparam


class DroneTrainBase:
    def __init__(self, run_name, ppo_params, drone_params, wrappers):
        self.run_name = run_name
        self.ppo_params = ppo_params
        self.drone_params = drone_params
        self.wrappers = wrappers or []

        self.env = None
        self.eval_env = None
        self.log_dir = None
        self.new_logger = None
        self.metrics_callback = None

    def make_vec_env(self, env_name):
        ppo_params = self.ppo_params
        drone_params = self.drone_params
        wrappers = self.wrappers
        run_name = self.run_name

        def _make(eval=False):
            env = gym.make(
                env_name,
                ip_address="127.0.0.1",
                image_shape=ppo_params["image_shape"],
                params=drone_params,
                run_name=run_name,
                eval=eval,
                camera_name=drone_params["camera_name"],
            )
            for wrapper_cls, kwargs in wrappers:
                env = wrapper_cls(env, **kwargs)
            return env

        self.env = VecTransposeImage(
            DummyVecEnv([lambda: _make(eval=False)])
        )
        self.env = VecNormalize(self.env, norm_obs=False, norm_reward=True, clip_reward=10.0)

        self.eval_env = VecTransposeImage(
            DummyVecEnv([lambda: Monitor(_make(eval=True))])
        )
        self.eval_env = VecNormalize(self.eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0)

    def init_logger(self):
        self.log_dir = f"./tb_logs/{self.run_name}"
        self.new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        os.makedirs(f"./models/{self.run_name}", exist_ok=True)
        os.makedirs(f"./models/{self.run_name}/images", exist_ok=True)
        self.metrics_callback = MetricsCallback()

    def make_model(self, extractor_class, extractor_kwargs=None):
        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            learning_rate=self.ppo_params["learning_rate"],
            batch_size=self.ppo_params["batch_size"],
            n_steps=self.ppo_params["n_steps"],
            n_epochs=self.ppo_params["n_epochs"],
            gamma=self.ppo_params["gamma"],
            gae_lambda=self.ppo_params["gae_lambda"],
            clip_range=self.ppo_params["clip_range"],
            ent_coef=self.ppo_params["ent_coef"],
            vf_coef=self.ppo_params["vf_coef"],
            max_grad_norm=self.ppo_params["max_grad_norm"],
            device="cuda",
            tensorboard_log="./tb_logs/",
            verbose=1,
            policy_kwargs={
                "features_extractor_class": extractor_class,
                "features_extractor_kwargs": extractor_kwargs or {},
            }
        )
        self.model.set_logger(self.new_logger)

    def save_params(self):
        wrapper_params = {
            wrapper_cls.__name__: kwargs
            for wrapper_cls, kwargs in self.wrappers
        }
        all_params = {**self.ppo_params, **self.drone_params, "wrappers": wrapper_params}
        all_params_clean = {k: to_hparam(v) for k, v in all_params.items()}
        with open(f"./models/{self.run_name}/params.json", "w") as fp:
            json.dump(all_params_clean, fp, indent=4)

    def train(self):
        eval_callback = EvalCallback(
            self.eval_env,
            n_eval_episodes=5,
            best_model_save_path=f"./models/{self.run_name}",
            log_path=f"./models/{self.run_name}",
            eval_freq=self.ppo_params["n_steps"],
        )

        self.save_params()

        try:
            self.model.learn(
                total_timesteps=self.ppo_params["total_timesteps"],
                callback=[eval_callback, self.metrics_callback]
            )
            self.model.save(f"./models/{self.run_name}/model_final")
        finally:
            self._move_tb_logs()
            self.model.save(f"./models/{self.run_name}/model_interrupted")


    def _move_tb_logs(self):
        for subdir in glob.glob(f"{self.log_dir}/*/"):
            for f in glob.glob(f"{subdir}events.out.tfevents.*"):
                shutil.move(f, self.log_dir)
            shutil.rmtree(subdir)

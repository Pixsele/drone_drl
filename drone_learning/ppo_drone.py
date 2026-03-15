from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym


def make_env():
    env = gym.make("airgym:airsim-drone-ppo-v0",
                   ip_address="127.0.0.1",
                   step_length=0.25,
                   image_shape=(84,84,1))
    check_env(env, warn=True)
    return env

if __name__ == "__main__":
    env = SubprocVecEnv([lambda: make_env() for _ in range(1)])
    env = VecTransposeImage(env)

    eval_env = VecTransposeImage(DummyVecEnv([lambda: Monitor(make_env())]))

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
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
        eval_freq=1000
    )

    model.learn(total_timesteps=30_000, callback=eval_callback)

    model.save("ppo_airsim_drone_policy")
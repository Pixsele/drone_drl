import sys
import os

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

import sim.reinforcement_learning.airgym

env = gym.make('airsim-drone-ppo-v0', ip_address='127.0.0.1', step_length=0.1, image_shape = (128, 128, 1))

check_env(env)
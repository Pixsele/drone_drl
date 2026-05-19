import random

import numpy as np
import gymnasium as gym
import torch
from gymnasium import ObservationWrapper
from depth_anything_v2.dpt import DepthAnythingV2

class RgbToDepthWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder='vits', output_size=(128, 128)):
        super().__init__(env)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.output_size = output_size
        self.model = DepthAnythingV2(**model_configs[encoder]).to('cuda').eval()
        self.model.load_state_dict(torch.load(
            rf'C:\Prog\drone_drl\depth_anything_v2\depth_anything_v2_{encoder}.pth',
            map_location='cpu'
        ))

        h, w = output_size
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8),
            **{k: v for k, v in env.observation_space.spaces.items() if k != "image"},
        })

    def observation(self, obs):
        with torch.no_grad():
            depth = self.model.infer_image(obs["image"])
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = (depth * 255).astype(np.uint8)

            h, w = self.output_size
            from PIL import Image
            depth = np.array(Image.fromarray(depth).resize((w, h)))
            depth = depth[..., None]

        return {"image": depth, **{k: v for k, v in obs.items() if k != "image"}}


class RandomShiftWrapper(gym.ObservationWrapper):
    def __init__(self, env, pad : int = 6):
        super().__init__(env)
        self.pad = pad

    def observation(self, obs):
        h, w, c = obs.shape

        obs_padded = np.pad(
            obs,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='edge'
        )

        top  = np.random.randint(0, 2 * self.pad)
        left = np.random.randint(0, 2 * self.pad)

        return obs_padded[top:top+h, left:left+w, :]


class SaltPepperWrapper(gym.ObservationWrapper):
    def __init__(self, env, prob : float = 0.02):
        super().__init__(env)
        self.prob = prob

    def observation(self, obs):
        obs = obs.copy()

        salt_mask = np.random.random(obs.shape[:2]) < self.prob / 2
        obs[salt_mask] = 255.0

        pepper_mask = np.random.random(obs.shape[:2]) < self.prob / 2
        obs[pepper_mask] = 0.0

        return obs


class CutWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_size: int = 5, max_cut_count: int = 2):
        super().__init__(env)
        self.max_size = max_size
        self.max_cut_count = max_cut_count

    def observation(self, obs):
        obs = obs.copy()
        h, w, c = obs.shape

        for _ in range(self.max_cut_count):
            size = random.randint(1, self.max_size)
            row  = random.randint(0, h - 1)
            col  = random.randint(0, w - 1)
            obs[row:min(row + size, h), col:min(col + size, w), :] = 0.0

        return obs

class GaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma : float = 0.02):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs):
        noise = np.random.normal(0, self.sigma, obs.shape)

        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
            obs = np.clip(obs + noise, 0, 1)
            return (obs * 255).astype(np.uint8)

        return np.clip(obs + noise, 0, 1)

class DepthQuantizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, levels=32):
        super().__init__(env)
        self.levels = levels

    def observation(self, obs):
        if obs.dtype == np.uint8:
            obs_f = obs.astype(np.float32) / 255.0
        else:
            obs_f = obs.copy()

        obs_q = np.round(obs_f * self.levels) / self.levels

        if obs.dtype == np.uint8:
            return (obs_q * 255).astype(np.uint8)

        return obs_q

class DistortionWrapper(gym.ObservationWrapper):
    def __init__(self, env, strength=0.05):
        super().__init__(env)
        self.strength = strength

    def observation(self, obs):
        h, w, c = obs.shape
        y, x = np.indices((h, w))

        x_c = (x - w / 2) / (w / 2)
        y_c = (y - h / 2) / (h / 2)

        r = np.sqrt(x_c**2 + y_c**2)

        factor = 1 + self.strength * (r**2)

        x_dist = x_c * factor
        y_dist = y_c * factor

        x_new = ((x_dist + 1) * w / 2).astype(np.int32)
        y_new = ((y_dist + 1) * h / 2).astype(np.int32)

        x_new = np.clip(x_new, 0, w - 1)
        y_new = np.clip(y_new, 0, h - 1)

        return obs[y_new, x_new]
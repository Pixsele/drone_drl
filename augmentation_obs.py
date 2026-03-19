import numpy as np
import gymnasium as gym

class RandomShiftWrapper(gym.ObservationWrapper):
    def __init__(self, env, pad : int = 4):
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




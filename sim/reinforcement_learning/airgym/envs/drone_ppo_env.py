import sim.setup_path
import sim.cosysairsim as airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gymnasium as gym
from gymnasium import spaces
from sim.reinforcement_learning.airgym.envs.airsim_env import AirSimEnv

class AirSimDronePPOEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.drone.takeoffAsync().join()


    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        vx = float(action[0]) * 3
        vy = float(action[1]) * 3
        vz = float(action[2]) * 2

        self.drone.moveByVelocityAsync(
            vx,
            vy,
            vz,
            0.2
        ).join()

    def _compute_reward(self):
        reward = 0.0
        target = np.array([15.0, 0.0, -5.0])

        pos = np.array([self.state["position"].x_val,
                        self.state["position"].y_val,
                        self.state["position"].z_val])

        last_pos = np.array([self.state["prev_position"].x_val,
                             self.state["prev_posiiton"].y_val,
                             self.state["prev_posiiton"].z_val])

        x,y,z  = pos

        vel = np.array([self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val])

        dist = np.linalg.norm(target - pos)
        last_dist = np.linalg.norm(last_pos - pos)
        speed = np.linalg.norm(vel)

        reward += -0.08 * dist

        direction_to_target = target - pos
        direction_to_target /= (np.linalg.norm(direction_to_target) + 1e-6)
        forward_component = np.dot(vel, direction_to_target)
        reward += 0.6 * forward_component

        lateral_speed = np.linalg.norm(vel - forward_component * direction_to_target)
        reward -= 0.2 * lateral_speed

        alignment = np.dot(vel / (speed + 1e-6), direction_to_target)
        reward += 0.5 * alignment

        if dist < last_dist:
            reward += 0.02

        if abs(dist - last_dist) < 0.01:
            reward -= 0.02

        done = False
        if self.state["collision"]:
            reward -= 25.0
            done = True
        if dist < 1.8:
            reward += 35.0
            done = True

        if z > -0.3:
            reward -= 5 * abs(1 - z)

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._setup_flight()
        obs = self._get_obs()

        return obs, {}


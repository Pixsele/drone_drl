import cv2
from sympy.physics.vector.printing import params

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

        self.step_count = 0
        self.trajectory_length = 0.0

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
            0, airsim.ImageType.DepthPerspective, True, False
        )

        self.params = {
            "target": (100.0,0.0,-5.0),
            "dist_reward": -0.01,
            "dir_to_target_reward": 0.3,
            "lateral_speed_reward": 0.015,
            "diff_dist_reward": 1.5,
            "diff_dist_not_fine": 0.02,
            "collision_fine": 20.0,
            "target_reward": 30.0,
            "vx" : 2.5,
            "vy" : 2.5,
            "vz" : 1.0,
            "max_steps": 500,        # максимум шагов за эпизод
            "stall_speed": 0.05,     # ниже этого считается зависанием
            "stall_steps": 30,
            "max_step_fine":5.0,
            "stall_fine": 10.0
        }

        self.stall_counter = 0

        target = np.array(self.params["target"])
        self.optimal_distance = np.linalg.norm(target)

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

        h, w, c = self.image_shape
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((w, h)).convert("L"))

        return im_final.reshape(self.image_shape)

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
        vx = float(action[0]) * self.params["vx"]
        vy = float(action[1]) * self.params["vy"]
        vz = float(action[2]) * self.params["vz"]

        self.drone.moveByVelocityAsync(
            vx,
            vy,
            vz,
            0.2
        ).join()

    def _compute_reward(self):
        reward = 0.0
        target = np.array([self.params["target"][0], self.params["target"][1], self.params["target"][2]])

        pos = np.array([self.state["position"].x_val,
                        self.state["position"].y_val,
                        self.state["position"].z_val])

        last_pos = np.array([self.state["prev_position"].x_val,
                             self.state["prev_position"].y_val,
                             self.state["prev_position"].z_val])

        x,y,z  = pos

        vel = np.array([self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val])

        dist = np.linalg.norm(target - pos)
        last_dist = np.linalg.norm(target - last_pos)
        speed = np.linalg.norm(vel)

        self.trajectory_length += np.linalg.norm(pos - last_pos)

        reward += self.params["dist_reward"] * dist

        direction_to_target = target - pos
        direction_to_target /= (np.linalg.norm(direction_to_target) + 1e-6)
        forward_component = np.dot(vel, direction_to_target)
        reward += self.params["dir_to_target_reward"] * forward_component

        lateral_speed = np.linalg.norm(vel - forward_component * direction_to_target)
        reward -= self.params["lateral_speed_reward"] * lateral_speed

        # alignment = np.dot(vel / (speed + 1e-6), direction_to_target)
        # reward += 0.5 * alignment

        if dist < last_dist:
            reward += (last_dist - dist) * self.params["diff_dist_reward"]

        if abs(dist - last_dist) < 0.005:
            reward -= self.params["diff_dist_not_fine"]

        done = False
        info = {}

        if self.state["collision"]:
            reward -= self.params["collision_fine"]
            done = True

            info["success"] = False
            info["collision"] = True
            info["stall"] = False

        if dist < 2.0:
            reward += self.params["target_reward"]
            done = True

            info["success"] = True
            info["collision"] = False
            info["stall"] = False
            info["steps_to_goal"] = self.step_count
            info["path_efficiency"] = (
                self.optimal_distance / self.trajectory_length
                if self.trajectory_length > 0 else 0.0
            )

        # if z > -0.5:
        #     reward -= 0.05 * abs(z + 5)

        return reward, done, info

    def _check_stall(self):
        vel = np.array([self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val])
        return bool(np.linalg.norm(vel) < 0.05)

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done, info = self._compute_reward()
        self.step_count += 1

        terminated = done
        truncated = False

        if self._check_stall():
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if not terminated:
            if self.step_count >= self.params["max_steps"]:
                truncated = True

                reward -= self.params["max_step_fine"]

                info["success"] = False
                info["collision"] = False
                info["stall"] = False

            elif self.stall_counter >= self.params["stall_steps"]:
                truncated = True

                reward -= self.params["stall_fine"]

                info["success"] = False
                info["collision"] = False
                info["stall"] = True

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._setup_flight()
        obs = self._get_obs()

        self.step_count = 0
        self.trajectory_length = 0.0
        self.stall_counter = 0

        return obs, {}


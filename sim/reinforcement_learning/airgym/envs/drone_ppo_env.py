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
    def __init__(self, ip_address, step_length, image_shape, drone_id=0):
        super().__init__(image_shape)

        self.step_length = step_length
        self.image_shape = image_shape

        self.step_count = 0
        self.trajectory_length = 0.0
        self.stall_counter = 0


        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        if drone_id == 0:
            self.drone_name = "SimpleFlight"
        else:
            self.drone_name = f"Drone{drone_id}"

        self.spawn_pose = airsim.Vector3r(0, 1.5 * drone_id, 0)

        self.client = airsim.MultirotorClient(ip=ip_address)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self._add_drone()
        self._setup_flight()

        if self.image_shape[2] == 1:
            self.image_request = airsim.ImageRequest(
                0, airsim.ImageType.DepthPerspective, True, False
            )

        if self.image_shape[2] == 3:
            self.image_request = airsim.ImageRequest(
                0, airsim.ImageType.Scene, False, False
            )


        self.params = {
            "target": (100.0,0.0,-5.0),         # цель
            "dist_reward": -0.01,               # коэф. штраф за дистанцию за шаг
            "dir_to_target_reward": 0.3,        # коэф. награда за движение в направлении к цели
            "lateral_speed_reward": 0.015,      # коэф. штрафа за боковую скорость
            "diff_dist_reward": 1.5,            # коэф. награды за движение
            "diff_dist_not_fine": 0.02,         # штраф за зависание
            "collision_fine": 20.0,             # штраф за столкновение
            "target_reward": 30.0,              # награда за достижение цели
            "vx" : 2.5,                         #
            "vy" : 2.5,                         #
            "vz" : 1.0,                         # 
            "max_steps": 500,                   # максимум шагов за эпизод
            "stall_speed": 0.05,                # ниже этого считается зависанием
            "stall_steps": 30,                  # шагов на зависание для завершения эпизода
            "max_step_fine":5.0,                # штраф за большое кол-во шагов
            "stall_fine": 10.0                  # штраф за полное зависание
        }


        target = np.array(self.params["target"])
        self.optimal_distance = np.linalg.norm(target)

    def __del__(self):
        self.client.reset()

    def _add_drone(self):
        if self.drone_name == "SimpleFlight":
            return

        if self.drone_name in self.client.listVehicles():
            return
        self.client.simAddVehicle(self.drone_name, "simpleflight", airsim.Pose(self.spawn_pose))

    def _setup_flight(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()

    def transform_obs(self, responses):
        response = responses[0]
        h, w, c = self.image_shape

        if self.image_shape[2] == 1:
            img1d = np.array(response.image_data_float, dtype=np.float32)

            img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
            img2d = np.reshape(img1d, (response.height, response.width))

            from PIL import Image
            image = Image.fromarray(img2d)
            image = image.resize((w, h)).convert("L")

            img = np.array(image, dtype=np.uint8)
            return img.reshape(h, w, 1)

        if self.image_shape[2] == 3:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            img_rgb = img1d.reshape(response.height, response.width, 3)

            from PIL import Image
            image = Image.fromarray(img_rgb)
            image = image.resize((w, h))

            img = np.array(image, dtype=np.uint8)
            return img.reshape(h, w, 3)
        return None

    def _get_obs(self):
        responses = self.client.simGetImages([self.image_request], vehicle_name=self.drone_name)
        image = self.transform_obs(responses)

        self.drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.client.simGetCollisionInfo(vehicle_name=self.drone_name).has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        vx = float(action[0]) * self.params["vx"]
        vy = float(action[1]) * self.params["vy"]
        vz = float(action[2]) * self.params["vz"]

        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            0.2,
            vehicle_name=self.drone_name
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

        self.client.simSetVehiclePose(
            airsim.Pose(self.spawn_pose, airsim.Quaternionr()),
            ignore_collision=True,
            vehicle_name=self.drone_name
        )

        self._setup_flight()

        self.step_count = 0
        self.trajectory_length = 0.0
        self.stall_counter = 0
        obs = self._get_obs()
        return obs, {}

import math

import sim.setup_path
import sim.cosysairsim as airsim
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from sim.reinforcement_learning.airgym.envs.airsim_env import AirSimEnv


class AirSimCarPPOEnv(AirSimEnv):
    def __init__(self, ip_address, step_length,image_shape, params, id=0):
        super().__init__(image_shape)

        self.step_length = step_length
        self.image_shape = image_shape
        self.params = params

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)

        #TODO
        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(2,),
            dtype=np.float32
        )

        if self.image_shape[2] == 1:
            self.image_request = airsim.ImageRequest(
                0, airsim.ImageType.DepthPerspective, True, False
            )

        if self.image_shape[2] == 3:
            self.image_request = airsim.ImageRequest(
                0, airsim.ImageType.Scene, False, False
            )

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        vx = float(action[0]) * self.params["vx"]
        vy = float(action[1]) * self.params["vy"]

        self.car_controls.throttle = vx
        self.car_controls.steering = vy

        self.car.setCarControls(self.car_controls)

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
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses)

        self.car_state = self.car.getCarState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.car_state.kinematics_estimated.position
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > THRESH_DIST:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        info = {}

        return reward, done, info

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done, info = self._compute_reward()

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._setup_car()
        return self._get_obs(), {}

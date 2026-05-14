import time

import cv2

import sim.cosysairsim as airsim
import numpy as np

from gymnasium import spaces

from drone_learning.logging_config import setup_logger
from sim.reinforcement_learning.airgym.envs.airsim_env import AirSimEnv


class DroneDirectionBaseEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape, params, run_name, camera_name = 0 ,eval=False, log_image=True, show_image=False):
        super().__init__(image_shape)

        self.logger = setup_logger(run_name=run_name)

        self.client = airsim.MultirotorClient(ip=ip_address)

        self.step_count = 0
        self.reset_count = 0

        self.image_shape = image_shape
        self.camera_name= camera_name

        self.log_image = log_image
        self.show_image = show_image

        self.params = params

        self.run_name = run_name
        self.eval = eval

        self.drone_name = "SimpleFlight"
        self.spawn_pose = airsim.Vector3r(0.0,0.0,-2.0)
        self.start_pos = self.spawn_pose

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.action_space = self._build_action_space()
        self.observation_space = self._build_obs_space()

        if self.image_shape[2] == 1:
            self.image_request = airsim.ImageRequest(
                self.camera_name, airsim.ImageType.DepthPerspective, True, False
            )

        if self.image_shape[2] == 3:
            self.image_request = airsim.ImageRequest(
                self.camera_name, airsim.ImageType.Scene, False, False
            )

        self._setup_flight()


    def __del__(self):
        self.client.reset()

    def _build_action_space(self):
        space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        return space

    def _build_obs_space(self):
        space =  spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=self.image_shape,
                dtype=np.uint8
            ),
            "direction": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            ),
        })
        return space

    def _setup_flight(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()

    def _get_image(self, responses):
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

    def _get_extra_obs(self):
        direction = np.array(self.params["direction"], dtype=np.float32)
        return {"direction": direction}

    def _get_obs(self):
        responses = self.client.simGetImages([self.image_request], vehicle_name=self.drone_name)
        image = self._get_image(responses)

        if self.log_image and self.run_name is not None and self.step_count % 1000 == 0:
            cv2.imwrite(f"./models/{self.run_name}/images/{self.step_count}.jpg", image)
            self.logger.debug(f"Image saved to ./models/{self.run_name}/images/{self.step_count}.jpg")

        if self.show_image:
            cv2.imshow(f"image_{self.step_count}", image)
            cv2.waitKey(1)

        self.drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = self.client.simGetCollisionInfo(vehicle_name=self.drone_name).has_collided
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state["collision"] = collision

        extra = self._get_extra_obs()
        return {"image": image, **extra}

    def _do_action(self, action):
        vx = float(action[0]) * self.params["vx"]
        vy = float(action[1]) * self.params["vy"]
        vz = float(action[2]) * self.params["vz"]

        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            0.2,
        ).join()

    def _compute_reward(self):
        direction = np.array(self.params["direction"], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        pos = np.array([self.state["position"].x_val,
                        self.state["position"].y_val,
                        self.state["position"].z_val])

        last_pos = np.array([self.state["prev_position"].x_val,
                             self.state["prev_position"].y_val,
                             self.state["prev_position"].z_val])

        reward = 0.0

        displacement = pos - last_pos
        progress = np.dot(displacement, direction)
        r_progress = self.params["progress_reward"] * progress

        lateral = displacement - np.dot(displacement, direction) * direction
        lateral_dist = np.linalg.norm(lateral)
        r_lateral = -self.params["lateral_penalty"] * lateral_dist

        z_drop = pos[2] - self.start_pos[2]
        r_altitude = 0.0
        if z_drop > 1.0:
            r_altitude = -self.params["altitude_penalty"] * z_drop

        reward += r_progress + r_lateral + r_altitude

        done = False
        info = {}
        r_collision = 0.0

        if self.state["collision"]:
            r_collision = -self.params["collision_fine"]
            reward += r_collision
            done = True
            info = {"success": False, "collision": True}

        self.logger.info(
            f"drone={self.drone_name} | "
            f"eval={self.eval} | "
            f"step={self.step_count:3d} | "
            f"progress={r_progress:+.3f} | "
            f"lateral={r_lateral:+.3f} | "
            f"altitude={r_altitude:+.3f} | "
            f"collision={r_collision:+.3f} | "
            f"total={reward:+.3f} | "
            f"dir={self.params['direction']} | "
            f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}] | "
            f"done={done} | "
            f"info=[{info}]"
        )

        return reward, done, info

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()

        reward, done, info = self._compute_reward()

        terminated = done
        truncated = False

        if not terminated:
            direction = np.array(self.params["direction"], dtype=np.float32)
            direction /= (np.linalg.norm(direction) + 1e-6)
            distance_traveled = np.dot(
                np.array([self.state["position"].x_val,
                          self.state["position"].y_val,
                          self.state["position"].z_val]) - self.start_pos,
                direction
            )
            if distance_traveled > self.params["target_distance"]:
                reward += self.params["target_reward"]
                terminated = True
                info["success"] = True
                info["collision"] = False

            if self.step_count >= self.params["max_steps"]:
                truncated = True
                info["success"] = False
                info["collision"] = False

        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.client.reset()

        self.client.simSetVehiclePose(
            airsim.Pose(self.spawn_pose, airsim.Quaternionr()),
            ignore_collision=True,
            vehicle_name=self.drone_name
        )

        time.sleep(0.2)
        self._setup_flight()

        self.reset_count += 1
        r = self.reset_count

        if not self.eval:
            if r < 50:
                angle = 0.0
            elif r < 150:
                angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            elif r < 350:
                angle = np.random.uniform(-np.pi / 3, np.pi / 3)
            else:
                angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            self.params["direction"] = [float(np.cos(angle)), float(np.sin(angle)), 0.0]

        self.logger.info(
            f"drone={self.drone_name} | "
            f"eval={self.eval} | "
            f"reset_count={self.reset_count:3d} | "
             f"direction={self.params.get('direction', 'N/A')} | "
        )

        self.step_count = 0
        obs = self._get_obs()

        self.start_pos = np.array([
            self.drone_state.kinematics_estimated.position.x_val,
            self.drone_state.kinematics_estimated.position.y_val,
            self.drone_state.kinematics_estimated.position.z_val,
        ])
        return obs, {}

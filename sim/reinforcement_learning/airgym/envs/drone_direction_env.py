import cv2

import sim.setup_path
import sim.cosysairsim as airsim
import numpy as np

from gymnasium import spaces
from sim.reinforcement_learning.airgym.envs.airsim_env import AirSimEnv

class AirSimDroneDirectionPPOEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, params, run_name, client_id=0):
        super().__init__(image_shape)

        self.current_step = 0
        self.reset_count = 0
        self.run_name = run_name

        self.step_length = step_length
        self.image_shape = image_shape
        self.params = params

        self.step_count = 0
        self.trajectory_length = 0.0
        self.stall_counter = 0


        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        if client_id == 0:
            self.drone_name = "SimpleFlight"
        else:
            self.drone_name = f"Drone{client_id}"

        if client_id == 0:
            vector = np.array([0.0, 0.0, -2])
            self.spawn_pose = airsim.Vector3r(0.0,0.0,-2.0)
            self.start_pose = vector
        else:
            #TODO
            self.spawn_pose = airsim.Vector3r(0, 11 * -client_id, -2)
        print(f"client_id: {client_id} - spawn_pose: {self.spawn_pose}")


        self.client = airsim.MultirotorClient(ip=ip_address)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=image_shape,
                dtype=np.uint8
            ),
            "direction": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            ),
        })

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

        if self.run_name is not None and self.current_step % 1000 == 0:
            cv2.imwrite(f"./models/{self.run_name}/images/{self.current_step}.jpg", image)

        direction = np.array(self.params["direction"], dtype=np.float32)

        self.drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        # print(f"{self.drone_name} - {self.drone_state.kinematics_estimated.position}")

        collision = self.client.simGetCollisionInfo(vehicle_name=self.drone_name).has_collided
        self.state["collision"] = collision

        return {"image": image, "direction": direction}

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
        direction = np.array(self.params["direction"], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        pos = np.array([self.state["position"].x_val,
                        self.state["position"].y_val,
                        self.state["position"].z_val])

        last_pos = np.array([self.state["prev_position"].x_val,
                             self.state["prev_position"].y_val,
                             self.state["prev_position"].z_val])

        x,y,z  = pos

        vel = np.array([self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val]
                       )

        reward = 0.0
        speed = np.linalg.norm(vel)
        self.trajectory_length += np.linalg.norm(pos - last_pos)


        alignment = np.dot(vel / (speed + 1e-6), direction)
        reward += self.params["alignment_reward"] * alignment * speed

        if alignment < 0.5:
            reward -= self.params["wrong_direction_penalty"] * (0.5 - alignment) * speed

        displacement = pos - last_pos
        progress = np.dot(displacement, direction)
        reward += self.params["progress_reward"] * progress * speed


        lateral_speed = np.linalg.norm(vel - np.dot(vel,direction) * direction)
        reward -= self.params["lateral_penalty"] * lateral_speed

        z_drop = pos[2] - self.start_pos[2]
        if z_drop > 0.3:
            reward -= self.params["altitude_penalty"] * z_drop

        done = False
        info = {}

        if self.state["collision"]:
            reward -= self.params["collision_fine"]
            done = True

            info = {
                "success": False,
                "collision": True,
                "stall": False,
            }

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
        self.current_step += 1

        terminated = done
        truncated = False

        if self._check_stall():
            self.stall_counter += 1
        else:
            self.stall_counter = 0

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
                info["stall"] = False

            if self.step_count >= self.params["max_steps"]:
                truncated = True
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
        self.reset_count += 1

        self.client.simSetVehiclePose(
            airsim.Pose(self.spawn_pose, airsim.Quaternionr()),
            ignore_collision=True,
            vehicle_name=self.drone_name
        )

        self._setup_flight()

        # if self.reset_count < 100:
        #     self.params["direction"] = [1.0, 0.0, 0.0]
        #
        # else:
        #     if self.reset_count % self.params["direction_rand_frequency"] == 0:
        #         angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        #         self.params["direction"] = [
        #             float(np.cos(angle)),
        #             float(np.sin(angle)),
        #             0.0,
        #         ]
        #         print(f"Direction: {self.params['direction']}")

        if self.reset_count < 100:
            self.params["direction"] = [1.0, 0.0, 0.0]

        elif self.reset_count < 200:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            self.params["direction"] = [
                float(np.cos(angle)),
                float(np.sin(angle)),
                0.0,
            ]
            print(f"Direction: {self.params['direction']}")

        elif self.reset_count < 400:
            angle = np.random.uniform(-np.pi / 3, np.pi / 3)
            self.params["direction"] = [
                float(np.cos(angle)),
                float(np.sin(angle)),
                0.0,
            ]
            print(f"Direction: {self.params['direction']}")

        else:
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            self.params["direction"] = [
                float(np.cos(angle)),
                float(np.sin(angle)),
                0.0,
            ]
            print(f"Direction: {self.params['direction']}")

        self.step_count = 0
        self.trajectory_length = 0.0
        self.stall_counter = 0
        obs = self._get_obs()

        self.start_pos = np.array([
            self.drone_state.kinematics_estimated.position.x_val,
            self.drone_state.kinematics_estimated.position.y_val,
            self.drone_state.kinematics_estimated.position.z_val,
        ])
        return obs, {}

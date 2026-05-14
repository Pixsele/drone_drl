import time

import cv2


import sim.setup_path
import sim.cosysairsim as airsim
import numpy as np

from gymnasium import spaces
from sim.reinforcement_learning.airgym.envs.airsim_env import AirSimEnv

class AirSimDroneDirectionPPOEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, params, run_name, eval=False,client_id=0):
        super().__init__(image_shape)

        self.phase_thresholds = [100, 300, 600]
        self.current_step = 0
        self.reset_count = 0
        self.run_name = run_name
        self.eval = eval

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
            info = {"success": False, "collision": True, "stall": False}

        # print(
        #     f"step={self.step_count:3d} | "
        #     f"progress={r_progress:+.3f} | "
        #     f"lateral={r_lateral:+.3f} | "
        #     f"altitude={r_altitude:+.3f} | "
        #     f"collision={r_collision:+.3f} | "
        #     f"total={reward:+.3f} | "
        #     f"dir={self.params['direction']} | "
        #     f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]"
        # )

        return reward, done, info

    def _check_stall(self):
        vel = np.array([self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val])
        return bool(np.linalg.norm(vel) < 0.05)

    def step(self, action):
        pose = self.client.simGetObjectPose("BP_qr_C_3")
        print(pose)

        pose = self.client.simGetObjectPose("BP_qr_C_4")
        print(pose)

        # objects = self.client.simListSceneObjects(".*")
        # print([o for o in objects if "qr" in o.lower()])

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

        self.client.reset()

        self.client.simSetVehiclePose(
            airsim.Pose(self.spawn_pose, airsim.Quaternionr()),
            ignore_collision=True,
            vehicle_name=self.drone_name
        )

        time.sleep(0.2)

        self._setup_flight()


        r = self.reset_count

        # if self.eval:
        #     angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        # else:
        #     if r < 50:
        #         angle = 0.0
        #     elif r < 150:
        #         angle = np.random.uniform(-np.pi / 6, np.pi / 6)
        #     elif r < 350:
        #         angle = np.random.uniform(-np.pi / 3, np.pi / 3)
        #     else:
        #         angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        #
        # self.params["direction"] = [float(np.cos(angle)), float(np.sin(angle)), 0.0]
        #
        # print(f"[{self.reset_count}] Direction {self.params['direction']}  angle: {np.degrees(angle):.1f}°")

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

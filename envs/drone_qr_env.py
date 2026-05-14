import numpy as np

from envs import DroneDirectionBaseEnv

from gymnasium import spaces

import sim.cosysairsim as airsim

from pyzbar import pyzbar

class DroneDirectionQREnv(DroneDirectionBaseEnv):
    def __init__(self, ip_address, image_shape, params, run_name, camera_name = 0 ,eval=False, log_image=True, show_image=False):
        super().__init__(ip_address, image_shape, params, run_name, camera_name, eval, log_image, show_image)
        self.spawn_pose = airsim.Vector3r(0.0,0.0,-10.0)
        self.start_pos = self.spawn_pose
        self.state["prev_delta_to_qr"] = np.zeros(3)
        self.state["last_qr_obs"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


    def _build_obs_space(self):
        space =  spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=self.image_shape,
                dtype=np.uint8
            ),
            "qr_pos": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            ),
        })
        return space

    def _detect_qr(self, image):
        decoded = pyzbar.decode(image)

        if not decoded:
            return None

        h,w,_ = image.shape

        qr = decoded[0]
        x, y, bw, bh = qr.rect

        cx_qr = x + bw / 2
        cy_qr = y + bh / 2

        dx = (cx_qr - w / 2) / (w / 2)
        dy = (cy_qr - h / 2) / (h / 2)
        bbox_size = (bw * bh) / (w * h)

        self.logger.info(decoded)
        return np.array([dx, dy, bbox_size, 1.0], dtype=np.float32)


    def _get_extra_obs(self):
        responses = self.client.simGetImages([self.image_request], vehicle_name=self.drone_name)
        image = self._get_image(responses)

        qr_obs = self._detect_qr(image)

        if qr_obs is not None:
            self.state["last_qr_obs"] = qr_obs
        else:
            self.state["last_qr_obs"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return {"qr_pos": self.state["last_qr_obs"]}

    def _compute_reward(self):
        qr_pose = self.client.simGetObjectPose("BP_Domain_QR0")
        qr_pos = np.array([
            qr_pose.position.x_val,
            qr_pose.position.y_val,
            qr_pose.position.z_val,
        ])

        drone_pos = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val,
            self.state["position"].z_val,
        ])

        delta = qr_pos - drone_pos
        dist = np.linalg.norm(delta)

        if self.state["prev_delta_to_qr"] is None:
            r_progress = 0.0
        else:
            prev_dist = np.linalg.norm(self.state["prev_delta_to_qr"])
            r_progress = (prev_dist - dist) * self.params.get("progress_reward", 5.0)

        self.state["prev_delta_to_qr"] = delta.copy()

        qr_visible = self.state["last_qr_obs"][3]

        r_dist = -max(0.0, dist - self.start_dist) * self.params["dist_penalty"]

        r_visible = self.params["visible_reward"] if qr_visible else -self.params["invisible_penalty"]

        reward = r_progress + r_dist + r_visible
        done = False
        info = {}

        #TODO collision Info
        if self.state["collision"]:
            if dist < self.params["land_threshold"]:
                reward += self.params["land_reward"]
                done = True
                info = {"success": True}
            else:
                reward -= self.params["collision_fine"]
                done = True
                info = {"success": False, "collision": True}

        self.logger.debug(
            f"step={self.step_count} | "
            f"dist={dist:.2f} | "
            f"qr_visible={qr_visible} | "
            f"progress={r_progress:+.3f} | "
            f"r_dist={r_dist:+.3f} | "
            f"r_visible={r_visible:+.3f} | "
            f"total={reward:+.3f} | "
            f"done={done} | info={info}"
        )

        return reward, done, info

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done, info = self._compute_reward()

        if self.step_count >= self.params["max_steps"]:
            truncated = True
            info.setdefault("success", False)
        else:
            truncated = False

        self.step_count += 1
        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        self.state["prev_delta_to_qr"] = None
        self.state["last_qr_obs"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        obs, info = super().reset(seed=seed, options=options)

        qr_pose = self.client.simGetObjectPose("BP_Domain_QR0")
        qr_pos = np.array([qr_pose.position.x_val, qr_pose.position.y_val, qr_pose.position.z_val])
        drone_pos = np.array([self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val])
        self.start_dist = np.linalg.norm(qr_pos - drone_pos)

        return obs, info

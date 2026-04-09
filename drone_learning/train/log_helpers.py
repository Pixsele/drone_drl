from stable_baselines3.common.callbacks import BaseCallback


def to_hparam(v):
    if isinstance(v, tuple):
        return str(v)
    if isinstance(v, bool):
        return v
    if hasattr(v, '__float__'):
        return float(v)
    return str(v)

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "success" not in info:
                continue

            self.episode_metrics.append({
                "success":          float(info.get("success", False)),
                "collision":        float(info.get("collision", False)),
                "stall":            float(info.get("stall", False)),
                "steps_to_goal":    info.get("steps_to_goal", None),
                "path_efficiency":  info.get("path_efficiency", None),
            })

            window = self.episode_metrics[-20:]

            self.logger.record("metrics/success_rate",
                sum(m["success"] for m in window) / len(window))
            self.logger.record("metrics/collision_rate",
                sum(m["collision"] for m in window) / len(window))
            self.logger.record("metrics/stall_rate",
                sum(m["stall"] for m in window) / len(window))

            steps = [m["steps_to_goal"] for m in window if m["steps_to_goal"]]
            if steps:
                self.logger.record("metrics/steps_to_goal", np.mean(steps))

            eff = [m["path_efficiency"] for m in window if m["path_efficiency"]]
            if eff:
                self.logger.record("metrics/path_efficiency", np.mean(eff))

        return True

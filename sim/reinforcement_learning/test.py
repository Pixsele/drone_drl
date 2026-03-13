import gymnasium as gym

env = gym.make("airgym:airsim-drone-ppo-v0", ip_address="127.0.0.1", step_length=0.25, image_shape=(84,84,1))
obs, _ = env.reset()
action = env.action_space.sample()
obs, r, term, trunc, info = env.step(action)
print("OK" if not term else "failed")
env.close()
import gymnasium as gym
"""
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
"""


class Game:
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.state, self.info = None, None
        self.reset()

    def reset(self):
        self.state, self.info = self.env.reset()

    def state(self):
        return self.state

    def act(self, action=None):
        if action is None:
            action = self.env.action_space.sample()

        return self.env.step(action)

    def show(self):
        self.env.render()


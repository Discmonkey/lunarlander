import gymnasium as gym
import numpy as np
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

State = np.ndarray
IsTerminated = bool
Reward = float


class Game:
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")

    def sample_action(self):
        return self.env.action_space.sample()

    def reset(self) -> State:
        state, _ = self.env.reset()
        return state

    def act(self, action) -> tuple[State, Reward, IsTerminated]:
        state, reward, done, _, _ = self.env.step(action)
        return state, reward, done

    def show(self):
        self.env.render()

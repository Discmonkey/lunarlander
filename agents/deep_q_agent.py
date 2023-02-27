from agents.agent import Agent

import random
import torch


class QNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.stack(x)


class Replay:
    def __init__(self, max_size=10000):
        self.pool = []
        self.max_size = max_size

    def add(self,
            state_from,
            state_to,
            action,
            reward):
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

        self.pool.append(
            (state_from, state_to, action, reward))

    def get_batch(self, n):
        if n > len(self.pool):
            return self.pool.copy()

        return random.sample(self.pool, n)


class DeepQAgent(Agent):

    def update(self, gamma, episode_count, step_count):
        pass

    def __init__(self):
        self._update_target_count = 0


    def act(self, game):
        pass

    def _preprocess(self, state):
        pass

    def _network(self):
        self._student = QNetwork()
        self._target = QNetwork()


    @staticmethod
    def new_agent() -> Agent:
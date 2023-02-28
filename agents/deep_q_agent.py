import numpy

from agents.agent import Agent

import random
import torch

import config
from game import Game, State


class QNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
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
            reward,
            finished):
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

        self.pool.append(
            (state_from, state_to, action, reward, finished))

    def get_batch(self, n):
        if n > len(self.pool):
            return self.pool.copy()

        return random.sample(self.pool, n)


class DeepQAgent(Agent):

    def __init__(self):
        self._update_target_count = 0
        self._gamma = config.gamma
        self._replay_buffer = Replay()
        self._normalizer = numpy.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5
        ])

        self._from_batch = numpy.zeros((32, 8), dtype=float)
        self._to_batch = numpy.zeros((32, 8), dtype=float)
        self._finished = numpy.zeros((32, 1), dtype=float)
        self._rewards = numpy.zeros((32, 1), dtype=float)

        self._student_network = QNetwork()
        self._target_network = QNetwork()

        self._loss = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._student_network.parameters(), lr=config.lr)

    def update(self, episode_count, step_count, state_0, action, state_1, reward, finished):
        self._replay_buffer.add(
            self._preprocess(state_0),
            self._preprocess(state_1),
            action, reward, finished)

        x, y = self._process_batch(self._replay_buffer.get_batch(32))

        self._optimizer.zero_grad()

        prediction = self._student_network(x)
        loss = self._loss(prediction, y)
        if config.clip_error:
            loss = torch.clamp(loss, -1, 1)

        loss.backward()
        self._optimizer.step()

        if step_count % config.update_target_every == 2:
            self._target_network.load_state_dict(self._student_network.state_dict())

    def evaluate(self, state: State) -> numpy.ndarray:
        return self._student_network(self._preprocess_batch(state)).to_numpy()

    def _preprocess(self, state: State):
        return state - self._normalizer

    def _process_batch(self, batch):
        for num, (from_, to, action, reward, finished) in enumerate(batch):
            self._from_batch[num] = from_
            self._to_batch[num] = to

            if finished:
                self._finished[num] = 1
                self._rewards[num] = reward
            else:
                self._finished[num] = 0

        x = torch.tensor(self._from_batch, dtype=torch.float)
        rewards = torch.Tensor(self._rewards)
        finished = torch.Tensor(self._finished)
        # 32 x 4
        y = self._target_network(torch.tensor(self._to_batch, dtype=torch.float))
        # 32 x 1
        y = torch.max(y, 1)[0]
        # 32 x 1
        y = torch.add(y, rewards)
        y = torch.where(finished > 0, rewards, y)

        return x, y

    def _create_training_batch(self):
        pass

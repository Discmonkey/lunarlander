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
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
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

        self._from_batch = numpy.zeros((config.mini_batch_size, 8), dtype=float)
        self._to_batch = numpy.zeros((config.mini_batch_size, 8), dtype=float)
        self._finished = numpy.zeros((config.mini_batch_size, 1), dtype=float)
        self._rewards = numpy.zeros((config.mini_batch_size, 1), dtype=float)

        self._student_network = QNetwork()
        self._target_network = QNetwork()

        self._loss = torch.nn.MSELoss()

    def update(self, state_0, action, state_1, reward, finished):
        if random.random() > .98:
            self._target_network.load_state_dict(self._student_network.state_dict())

        self._replay_buffer.add(
            self._preprocess(state_0),
            self._preprocess(state_1),
            action, reward, finished)

    def train(self, num_steps=1):
        optimizer = torch.optim.Adam(self._student_network.parameters(), lr=config.lr)

        for step in range(num_steps):
            x, y, actions = self._process_batch(self._replay_buffer.get_batch(config.mini_batch_size))
            optimizer.zero_grad()

            prediction = self._student_network(x).gather(1, actions)
            loss = self._loss(prediction, y)
            if config.clip_error:
                loss = torch.clamp(loss, -1, 1)
            loss.backward()
            optimizer.step()

            if step % config.print_step == 1:
                print(f"{step} - {loss}")

    def evaluate(self, state: State) -> numpy.ndarray:
        torch_state = torch.tensor(self._preprocess(state), dtype=torch.float)
        prediction = self._student_network(torch_state)
        return prediction.detach().numpy()

    def _preprocess(self, state: State):
        return state - self._normalizer

    def _process_batch(self, batch):
        actions = numpy.zeros((len(batch), 1))
        for num, (from_, to, action, reward, finished) in enumerate(batch):
            self._from_batch[num] = from_
            self._to_batch[num] = to
            actions[num][0] = action
            self._rewards[num] = reward
            self._finished[num] = 1 if finished else 0

        x = torch.tensor(self._from_batch[:len(batch)], dtype=torch.float)
        rewards = torch.Tensor(self._rewards[:len(batch)])
        finished = torch.Tensor(self._finished[:len(batch)])
        # 32 x 4
        y = self._target_network(torch.tensor(self._to_batch[:len(batch)], dtype=torch.float))
        # 32 x 1
        y = torch.max(y, 1)[0]
        # 32 x 1
        y = config.gamma * y
        y = torch.add(torch.unsqueeze(y, 1), rewards)
        y = torch.where(finished > 0, rewards, y)

        return x, y, torch.tensor(actions, dtype=torch.int64)

    def _create_training_batch(self):
        pass

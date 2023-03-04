from game import State

import numpy


class Agent:

    def evaluate(self, state: State) -> numpy.ndarray:
        raise NotImplementedError

    def update(self, state_0, action, state_1, reward, finished):
        raise NotImplementedError

    def train(self, num_steps=1):
        pass

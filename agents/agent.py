from game import State

import numpy


class Agent:

    def evaluate(self, state: State) -> numpy.ndarray:
        raise NotImplementedError

    def update(self, episode_count, step_count, state_0, state_1, reward, finished):
        raise NotImplementedError

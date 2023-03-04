import numpy as np

import config

import random

from agents.agent import Agent
from game import Game


def episode_epsilon(episode_num, min_eps=0.01) -> float:
    ratio = 1 - episode_num / config.epsilon_update_schedule
    discounted_epsilon = config.starting_epsilon * ratio
    return max(discounted_epsilon, min_eps)


def choose_random_action(epsilon) -> bool:
    return epsilon > random.random()


def sim_game(agent: Agent, render=True):
    game = Game(render=render)
    state = game.reset()
    total_reward = 0
    for step in range(1000):
        action = np.argmax(agent.evaluate(state))
        state, reward, done = game.act(action)
        total_reward += reward
        if done:
            return total_reward

    return total_reward

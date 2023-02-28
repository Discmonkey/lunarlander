import config

import random


def episode_epsilon(episode_num, min_eps=0.01) -> float:
    ratio = 1 - episode_num / config.epsilon_update_schedule
    discounted_epsilon = config.starting_epsilon * ratio
    return max(discounted_epsilon, min_eps)


def choose_random_action(epsilon) -> bool:
    return epsilon > random.random()


# This is a sample Python script.
import config
from game import Game
from agents.agent import Agent
import numpy as np

import utils

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def run():
    game = Game()
    agent = Agent()
    for episode in range(1000):
        state_0 = game.reset()
        eps = utils.episode_epsilon(episode)

        for step in range(config.max_steps):
            if utils.choose_random_action(eps):
                action = game.sample_action()
            else:
                action = np.argmax(agent.evaluate(state_0))

            state_1, reward, finished = game.act(action)
            agent.update(
                episode, step, state_0, state_1, reward, finished)

            # clean up for next episode
            if finished:
                break
            if episode % config.show_every == 2:
                game.show()
            state_0 = state_1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

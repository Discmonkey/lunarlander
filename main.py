# This is a sample Python script.
import config
from agents.deep_q_agent import DeepQAgent
from game import Game
from agents.agent import Agent
import numpy as np

import utils

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def run():
    game = Game()
    agent = DeepQAgent()
    for episode in range(10000):
        print(f"Starting episode {episode}")
        state_0 = game.reset()
        eps = utils.episode_epsilon(episode)
        total_reward = 0
        for step in range(config.max_steps):
            if utils.choose_random_action(eps):
                action = game.sample_action()
            else:
                evaluations = agent.evaluate(state_0)
                action = np.argmax(evaluations)

            state_1, reward, finished = game.act(action)
            total_reward += reward
            agent.update(state_0, action, state_1, reward, finished)
            agent.train()
            # clean up for next episode
            if finished:
                break

            state_0 = state_1
        if episode % config.show_every == 2:
            print(f"Total reward, no randomness: {utils.sim_game(agent, False)}")
        agent.train(config.train_length)
        print(f"Finished episode {episode}, total reward {total_reward}, with epsilon {eps}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

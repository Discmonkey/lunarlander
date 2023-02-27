class Agent:

    def act(self, game):
        raise NotImplementedError

    def update(self, gamma, episode_count, step_count):
        raise NotImplementedError

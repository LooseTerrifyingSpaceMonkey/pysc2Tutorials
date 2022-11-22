from reinforcelearn_raw_pysc2.base_agent import Agent
import random


class RandomAgent(Agent):
    def __int__(self):
        super(RandomAgent, self).__init__()
        self.name = 'RandomAgent'

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)
import random

import gym

import gym_runner
from agent_base import AgentBase

env = gym.make('FrozenLake-v0')

action_space = env.action_space


class Agent(AgentBase):

    def __init__(self):
        pass

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

    def act(self, state) -> int:
        return random.randint(0, action_space.n - 1)

    def learn(self, *args, **kwargs):
        pass


agent = Agent()

if __name__ == "__main__":
    gym_runner.run(env, agent)

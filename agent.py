import random
import numpy as np
import os
from gym import make
from train import transform_state

# def transform_state(state):
#     state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
#     result = []
#     result.extend(state)
#     return np.array(result)

class Agent:
    def __init__(self):
        file = np.load(__file__[:-8] + "/agent.npz")
        self.weight = file[file.files[0]]
        self.bias = file[file.files[1]]

    def act(self, state):
        return np.argmax(self.weight.dot(transform_state(state)) + self.bias)

    def reset(self):
        pass

if __name__ == "__main__":
    new_env = make("MountainCar-v0")
    agent = Agent()
    new_env.seed(9)
    reward = []
    weight = agent.weight
    b = agent.bias
    for i in range(10):
        state = new_env.reset()
        local_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, r, done, _ = new_env.step(action)
            local_reward += r
        reward.append(local_reward)
    print("max reward: ", np.max(reward))
    print("mean reward: ", np.mean(reward))
    print(reward)
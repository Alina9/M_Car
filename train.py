from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random

N_STEP = 3
GAMMA = 0.96

def go(weight, b):
    new_env = make("MountainCar-v0")
    new_env.seed(9)
    reward = []
    for i in range(20):
        state = new_env.reset()
        local_reward = 0
        done = False
        while not done:
            state = transform_state(state)
            action = np.argmax(weight@state + b)
            state, r, done, _ = new_env.step(action)
            local_reward += r
        reward.append(local_reward)
    return reward


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result)


class AQL:
    def __init__(self, state_dim, action_dim, alpha, eps):
        self.gamma = GAMMA ** N_STEP
        self.eps = eps
        self.weight = np.random.normal(scale=0.5, size=(action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.alpha = alpha

    def update(self, transition):
        state, action, next_state, reward, done = transition

        Q = self.weight @ state +self.b
        Qcur = Q[action]

        Q = self.weight @ next_state + self.b
        Qnext = Q.max()

        diff = reward + self.gamma * Qnext - Qcur
        self.weight[action] += self.alpha * diff * state
        self.b[action] = self.b[action] + self.alpha * diff

    def act(self, state, target=False):
        return np.argmax(self.weight@state + self.b)

    def save(self, path):
        weight = np.array(self.weight)
        b = np.array(self.b)
        np.savez("agent.npz", weight, b)


if __name__ == "__main__":
    env = make("MountainCar-v0")
    np.random.seed(9)
    env.seed(9)
    aql = AQL(state_dim=2, action_dim=3, alpha = 0.05, eps = 1)
    eps = aql.eps
    episodes = 1000
    old_revard = -2000
    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            reward += 20 * (abs(next_state[1]))
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        # if i % 20 == 0:
        #     aql.save("agent.npz")

        eps *= 0.7
        if old_revard < total_reward:
            r = go(aql.weight, aql.b)
            mean_revard = np.mean(r)
            if mean_revard > old_revard:
                old_revard = mean_revard
                aql.save("agent.npz")
                if (i+1) % 50 != 0:
                    print(f"episode: {i+1}, max revard: {np.max(r)}, mean revard: {mean_revard}")

        if (i+1) % 50 == 0:
            print(f"episode: {i+1}, revard: {total_reward}")
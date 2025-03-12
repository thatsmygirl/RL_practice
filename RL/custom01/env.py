from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
import gymnasium as gym


class CustomEnv(Env):
    def __init__(self):
        self.action_space = [i for i in range(-2, 3)]
        self.state = np.random.choice([-20, 0, 20, 40, 60, 80])
        self.prev_state = self.state
        self.episode_length = 1000

    def reset(self):
        self.state = np.random.choice([-20, 0, 20, 40, 60, 80])
        self.prev_state = self.state
        self.episode_length = 1000
        return self.state


    def step(self, action):
        self.episode_length -= 1
        self.state += self.action_space[action] ## discrete와 space.n 작용 다시보기

        if self.state >= 20 and self.state <= 25:
            reward = 100
        else:
            reward = -100


        prev_diff = min(abs(self.prev_state - 20), abs(self.prev_state - 25))
        curr_diff = min(abs(self.state - 20), abs(self.state - 25))

        if curr_diff <= prev_diff:
            if reward != 100:
                reward = reward + 50
            else:
                reward = 100
        else:
                reward = reward - 50


        self.prev_state = self.state

        if self.episode_length <= 0:
            done = True
        else:
            done = False

        return self.state, reward, done




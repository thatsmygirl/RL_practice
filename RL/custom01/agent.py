import numpy as np
import random
from collections import defaultdict

class QlearningAgent():
    def __init__(self, actions):
        self.action = actions # [i for i in range(-2, 3)]
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.01
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.action))
        else:
            q_list = self.q_table[state]
            action = self.arg_max(q_list)
        return action

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)

    def arg_max(self, q_list):
        max_idx_list = np.argwhere(q_list == np.amax(q_list))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)



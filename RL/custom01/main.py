from RL.custom01.env import CustomEnv
from RL.custom01.agent import QlearningAgent
import matplotlib.pyplot as plt

if __name__ =='__main__':
    env = CustomEnv()
    agent = QlearningAgent(actions=env.action_space)
    total_reward_list = []

    for episode in range(1000):
        state = env.reset()
        print('episode:', episode)

        total_reward = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            total_reward += reward

            if done:
                break

            agent.decay_epsilon()
            print('state', state, 'reward:', reward)
        total_reward_list.append(total_reward)

        print('total reward:', total_reward)
    plt.plot(total_reward_list)
    plt.show()
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1").unwrapped

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion() # interactive mode on -> plt.show()를 호출하지 않아도 그래프가 그려짐

device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        # deque는 양방향 queue를 의미한다.
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # memory로부터 batch_size 길이 만큼의 list를 반환한다.
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# Capacity (즉, maximum length) 10000짜리 deque 이다.
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    # random.random() >> [0.0, 1.0) 구간의 소수점 숫자를 반환한다.
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # 기댓값이 더 큰 action을 고르자.
            # 바로 예를 들어 설명해보면, 아래의 논리로 코드가 진행된다.
            '''
            policy_net(state) >> tensor([[0.5598, 0.0144]])
            policy_net(state).max(1) >> ('max value', 'max 값의 index')
            policy_net(state).max(1)[1] >> index를 선택함.
            policy_net(state).max(1)[1].view(1, 1) >> tensor([[0]]) 
            '''
            # 즉, 위 예제의 경우 index 0에 해당하는 action을 선택하는 것이다.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


'''
핵심 부분(학습 루프)
next state가 없는 경우를 분리해서 생각해야함
'''
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # 여기서부터는 memory의 길이 (크기)가 BATCH_SIZE 이상인 경우이다.
    # BATCH_SIZE의 크기만큼 sampling을 진행한다.
    transitions = memory.sample(BATCH_SIZE)

    # Remind) Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    # 아래의 코드를 통해 batch에는 각 항목 별로 BATCH_SIZE 개수 만큼의 성분이 한번에 묶여 저장된다.
    batch = Transition(*zip(*transitions))

    # 우선 lambda가 포함된 line의 빠른 이해를 위해 다음의 예제를 보자.
    # list(map(lambda x: x ** 2, range(5))) >> [0, 1, 4, 9, 16]
    '''
        즉, 아래의 line을 통해 BATCH_SIZE 개의 원소를 가진 tensor가 구성된다.  
        또한 각 원소는 True와 False 로 구성되어 있다. 
        batch.next_state는 다음 state 값을 가지고 있는 tensor로 크게 두 부류로 구성된다.
        >> None 혹은 torch.Size([1, 3, 40, 90]) 의 형태
        '''
    # 정리하면 아래의 코드는 batch.next_state에서 None을 갖는 원소를 False로,
    # 그렇지 않으면 True를 matching 시키는 line이다.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    # batch.next_state의 원소들 중 next state가 None이 아닌 원소들의 집합이다.
    # torch.Size(['next_state가 None이 아닌 원소의 개수', 3, 40, 90])의 형태
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    # 아래 세 변수의 size는 모두 torch.Size([128, 3, 40, 90]) 이다.
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
    # action_batch 에 들어있는 0 혹은 1 값으로 index를 설정하여 결과값에서 가져온다.
    # 즉, action_batch 값에 해당하는 결과 값을 불러온다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 한편, non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.

    # 일단 모두 0 값을 갖도록 한다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # non_final_mask에서 True 값을 가졌던 원소에만 값을 넣을 것이고, False 였던 원소에게는 0 값을 유지할 것이다.
    # target_net(non_final_next_states).max(1)[0].detach() 를 하면,
    # True 값을 갖는 원소의 개수만큼 max value 값이 모인다.
    # 이들을 True 값의 index 위치에만 반영시키도록 하자.
    # 정리하면 한 state에서 더 큰 action을 선택한 것에 대한 value 값이 담기게 된다.
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # expected Q value를 계산하자.
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


######################################################################
#
# 아래에서 주요 학습 루프를 찾을 수 있습니다. 처음으로 환경을
# 재설정하고 초기 ``state`` Tensor를 얻습니다. 그런 다음 행동을
# 샘플링하고, 그것을 실행하고, 다음 상태와 보상(항상 1)을 관찰하고,
# 모델을 한 번 최적화합니다. 에피소드가 끝나면 (모델이 실패)
# 루프를 다시 시작합니다.
#
# 아래에서 `num_episodes` 는 GPU를 사용할 수 있는 경우 600으로,
# 그렇지 않은 경우 50개의 에피소드를 설정하여 학습이 너무 오래 걸리지는 않습니다.
# 하지만 50개의 에피소드만으로는 CartPole에서 좋은 성능을 관찰하기에는 충분치 않습니다.
# 600개의 학습 에피소드 내에서 모델이 지속적으로 500개의 스텝을 달성하는 것을
# 볼 수 있어야 합니다. RL 에이전트 학습 과정에는 노이즈가 많을 수 있으므로,
# 수렴(convergence)이 관찰되지 않으면 학습을 재시작하는 것이 더 나은 결과를 얻을 수 있습니다.
#

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # 여기서 사용한 count()는 from itertools import count 로 import 한 것이다.
    # t -> 0, 1, 2, ... 의 순서로 진행된다.
    for t in count():

        # state shape >> torch.Size([1, 3, 40, 90])
        # action result >> tensor([[0]]) or tensor([[1]])
        action = select_action(state)

        # 선택한 action을 대입하여 reward와 done을 얻어낸다.
        # env.step(action.item())의 예시
        # >> (array([-0.008956, -0.160571,  0.005936,  0.302326]), 1.0, False, {})
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        optimize_model()

        # 목표 네트워크의 가중치를 소프트 업데이트
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

######################################################################
# 다음은 전체 결과 데이터 흐름을 보여주는 다이어그램입니다.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# 행동은 무작위 또는 정책에 따라 선택되어, gym 환경에서 다음 단계 샘플을 가져옵니다.
# 결과를 재현 메모리에 저장하고 모든 반복에서 최적화 단계를 실행합니다.
# 최적화는 재현 메모리에서 무작위 배치를 선택하여 새 정책을 학습합니다.
# "이전"의 target_net은 최적화에서 기대 Q 값을 계산하는 데에도 사용됩니다.
# 목표 네트워크 가중치의 소프트 업데이트는 매 단계(step)마다 수행됩니다.
#
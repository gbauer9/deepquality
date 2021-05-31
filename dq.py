from typing import final
import gym
import torch
import random
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple, deque

# TODO - Make list of best target policies and pick best one as final net
# TODO - Add play() function to play through game and get average reward
# TODO - Plot results for different hyperparameters

# Hyperparameters
HIDDEN_SIZE = 64
N_EPISODES = 100
BATCH_SIZE = 128
EPS = 0.9
EPS_DECAY = 200
EPS_MIN = 0.05
GAMMA = 0.999
LEARN_RATE = 0.001
TARGET_UPDATE = 10
MEM_CAP = 100000

# Define replay memory
Trans = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

class Memory():
    def __init__(self, size):
        self.mem = deque([], maxlen=size)

    def push(self, *args):
        self.mem.append(Trans(*args))

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return f"<Replay Memory {len(self.mem)}>"

# Define Neural Network
class QNet(nn.Module):
    def __init__(self, inputs, outputs):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(inputs, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Decide whether to play a random action or choose based on model
def getAction(model, state, actions):
    global steps_done
    # Exponential decay function: https://en.wikipedia.org/wiki/Exponential_decay
    # Add to EPS_MIN so it converges to EPS_MIN instead of 0
    eps_threshold = EPS_MIN + (EPS - EPS_MIN) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        if random.random() > eps_threshold:
            return int(model(state).argmax())
        else:
            return random.randrange(actions)

# Train
def train(model, episodes=N_EPISODES, mem_cap=MEM_CAP):
    # Play n number of games
    for e in range(episodes):
        total = 0
        # Reset state each game
        state = torch.FloatTensor([env.reset()])
        # Play until done
        while True:
            # Pick a move from the action space and take that action
            action = getAction(model, state, env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor([next_state])

            total += reward

            if done:
                reward = -1

            replay.push(state, torch.LongTensor([[action]]), torch.FloatTensor([[reward]]), next_state)

            learn(model)

            state = next_state

            if done:
                break
        
        if e % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())
        print(f"Episode {e}\nReward: {total}")

    return

def learn(model):
    # Don't do anything if there's not enough replay memory
    if len(replay) < BATCH_SIZE:
        return
    
    # Get batch of BATCH_SIZE from replay memory by random sample
    state_b, action_b, reward_b, n_state_b = zip(*replay.sample(BATCH_SIZE))

    state_b = torch.cat(state_b)
    action_b = torch.cat(action_b)
    reward_b = torch.cat(reward_b)
    n_state_b = torch.cat(n_state_b)

    final_mask = reward_b == -1.
    non_final_mask = reward_b != -1.

    q_cur = policy(state_b).gather(1, action_b)
    q_next = target(n_state_b).detach().max(1)[0].reshape(-1, 1)

    expected_q = torch.zeros(reward_b.shape)
    expected_q[final_mask] = reward_b[final_mask]
    expected_q[non_final_mask] = reward_b[non_final_mask] + (GAMMA * q_next[non_final_mask])

    loss = F.smooth_l1_loss(q_cur, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return
    
# Set up env
env = gym.make('CartPole-v0')
policy = QNet(4, 2)
target = QNet(4, 2)
target.load_state_dict(policy.state_dict())
replay = Memory(MEM_CAP)
optimizer = optim.Adam(policy.parameters(), LEARN_RATE)
steps_done = 0

train(policy)

total_reward = 0
state = torch.FloatTensor([env.reset()])
while True:
    env.render()
    next_state, reward, done, _ = env.step(int(target(state).argmax()))
    total_reward += reward
    next_state = torch.FloatTensor([next_state])
    state = next_state
    if done:
        break
env.close()

print(f"TOTAL REWARD: {total_reward}")
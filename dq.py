import gym
import torch
import random
import torch.nn as nn
from collections import namedtuple, deque

# TODO - Initialize environment and set possible actions, states, rewards
# TODO - Create replay memory consisting of state, action, reward, return state
# TODO - Initialize Neural Net to take state as input and return possible actions as output
# TODO - Loop through N episodes of the game, each time taking either random action or max estimated Q value
# based on exploration rate. Play that action and record the state, action, reward, new state in our 
# replay memory. Randomly sample mini-batch from the replay memory and set the target value y to either the 
# reward value for done state or best guess based on current estimation of Q(s,a) = (r + dis * max(Q(s', a'))).
# To do this, we make a separate target model that is updated with the weights of the training model every M steps

ACTION = {
    0 : 'Left',
    1 : 'Right'
}

# Hyperparameters

N_EPISODES = 50

EX_RATE = 1
EX_DECAY = 0.9975

UPDATE_TAR = 10

# Define replay memory

Trans = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

class Memory():
    def __init__(self, size):
        self.mem = deque([], maxlen=size)

    def push(self, *args):
        self.mem.append(Trans(*args))

    def sample(self):
        return random.sample(self.mem, 1)

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return f"<Memory {len(self.mem)}>"

# Define Neural Network

class QNet(nn.Module):
    def __init__(self, inputs, outputs):
        super(QNet, self).__init__()
        return

# Train

def train():

    env = gym.make('CartPole-v0')
    env.reset()

    replay = Memory(100)

    for _ in range(3):
        cur_s = tuple(env.env.state)
        a = env.action_space.sample()
        next_s, r, _, _ = env.step(a)
        replay.push(cur_s, a, r, next_s)
    env.close()

    print(replay.sample())


if __name__ == "__main__":
    train()
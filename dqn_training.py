from dqn_network import DQN
from dqn_replay import ReplayBuffer
from dqn_utils import epsilon_greedy, plot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import numpy as np

class training():
    def __init__(self, environment, Variable, USE_CUDA, gamma = 0.99):
        self.environment = environment
        self.Variable = Variable
        self.model = DQN(self.environment.observation_space.shape[0], 
                        self.environment.action_space.n, environment=self.environment, 
                        Variable=self.Variable)
        def CUDA():
            if USE_CUDA:
                self.model = self.model.cuda()
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer(1000)

        self.gamma = gamma

    
    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        q_values      = self.model(state)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
    
        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss
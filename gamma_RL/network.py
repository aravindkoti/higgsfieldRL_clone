import torch
import torch.nn as nn
import random
import torch.autograd as autograd
import numpy as np

from dqn_utils import seed_everything



class gamma_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, environment, device, Variable, seed_number):
        self.environment = environment
        self.Variable = Variable
        self.device = device

        def seed(seed_number):
            seed_everything()
        

        super(gamma_DQN, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(random.uniform(0.5,1)), requires_grad=True)
        
        self.layers = nn.Sequential(
            nn.Linear(self.environment.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.environment.action_space.n)
        )
        
    def forward(self, x):
        return self.gamma * self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.environment.action_space.n)
        return action


#Going to implement a separate nn for gamma in this model
class gamma_DQN_epsilonseed(nn.Module):
    def __init__(self, num_inputs, num_actions, environment, device, Variable, gamma):
        self.environment = environment
        self.Variable = Variable
        self.device = device
        

        super(gamma_DQN_epsilonseed, self).__init__()
        # self.gamma = nn.Parameter(torch.tensor(random.uniform(0.5,1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        
        self.layers = nn.Sequential(
            nn.Linear(self.environment.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.environment.action_space.n)
        )
        
    def forward(self, x):
        return self.gamma * self.layers(x)
    
    def act(self, state, epsilon):

        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.environment.action_space.n)
        return action
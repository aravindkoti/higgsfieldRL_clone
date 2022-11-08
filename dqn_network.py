import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, environment, Variable):
        self.environment = environment
        self.Variable = Variable

        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(self.environment.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.environment.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.environment.action_space.n)
        return action
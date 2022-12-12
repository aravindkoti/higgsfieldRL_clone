from gamma_RL.network import gamma_DQN, gamma_DQN_epsilonseed
from dqn_replay import ReplayBuffer
from dqn_utils import epsilon_greedy, plot, seed_everything

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import random

import numpy as np
import wandb


class gamma_train():
    def __init__(self, environment, Variable, USE_CUDA, device, seed_number, gamma = 0.99):
        
        def seed(seed_number):
            seed_everything()

        self.environment = environment
        self.Variable = Variable
        self.device = device
        self.model = gamma_DQN(self.environment.observation_space.shape[0], 
                        self.environment.action_space.n, environment=self.environment,
                        device=self.device, Variable=self.Variable, seed_number=seed_number)
        def CUDA():
            if USE_CUDA:
                self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer(1000)

        self.gamma = self.model.gamma

        self.losses = []
        self.all_rewards = []

    
    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        #Gamma now included in network forward pass, so we need to divide out gamma for current
        #state
        q_values      = self.model(state) / (self.model.gamma)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]

        #Training gamma Model might not work because gamma is not included in forward pass
        #PROBLEM: NN is meant to estimate Q without gamma. Maybe separate NN to estimate gamma?
        #Actor-critic model?
        expected_q_value = reward + next_q_value * (1 - done)
        #self.gamma = self.model.gamma
    
        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            torch.clamp(self.model.gamma, min=0.0, max=1.0)
    
        return loss

    def training_loop(self, num_frames, batch_size, tensorboard = False, writer=None, 
                        run_number = 1, wandb_plot=False):

        if wandb_plot:
            wandb.init(
                project= "Gamma Cartpole Training",
                name= f"Experiment_{run_number}",
                config={
                    "environment": "CartPole-v0"
                }
            )

        episode_reward = 0

        state = self.environment.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon_instantiate = epsilon_greedy()
            epsilon = epsilon_instantiate.epsilon_by_frame(frame_idx)
            action = self.model.act(state, epsilon)
            
    
            next_state, reward, done, _ = self.environment.step(torch.tensor([[action]]).item())
            self.replay_buffer.push(state, action, reward, next_state, done)
    
            state = next_state
            episode_reward += reward
    
            if done:
                state = self.environment.reset()
                self.all_rewards.append(episode_reward)

                if wandb_plot:
                    wandb.log({"Rewards": episode_reward})
                    wandb.log({"Gamma": self.model.gamma.data})

                if tensorboard:
                    writer.add_scalar(f'Episode rewards run {run_number}', episode_reward, frame_idx)
                    writer.add_scalar(f'Gamma run {run_number}', self.gamma, frame_idx)

                episode_reward = 0
        
            if len(self.replay_buffer) > batch_size:
                loss = self.compute_td_loss(batch_size)
                self.losses.append(loss.data)

                if wandb_plot:
                    wandb.log({"Episode Losses": loss.data})
                    
                if tensorboard:
                    writer.add_scalar(f'Episode Losses run {run_number}',loss.data, frame_idx)
        
            if frame_idx % 200 == 0:
                plot(frame_idx, self.all_rewards, self.losses)

#This is a model which matches the random processes between runs, so we can isolate the qualitative differences between gamma values
class gamma_train_epsilonseed():
    def __init__(self, environment, Variable, USE_CUDA, device, gamma = 0.99):
      
        self.environment = environment
        self.Variable = Variable
        self.device = device
        self.model = gamma_DQN_epsilonseed(self.environment.observation_space.shape[0], 
                        self.environment.action_space.n, environment=self.environment,
                        device=self.device, Variable=self.Variable, gamma=gamma)
        def CUDA():
            if USE_CUDA:
                self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam([
                                        {'params': self.model.gamma, 'lr': 0.0015},
                                        {'params': self.model.layers.parameters(), 'lr': 0.001}
                                        ], lr=0.01)
        self.replay_buffer = ReplayBuffer(1000)

        self.gamma = self.model.gamma

        self.losses = []
        self.all_rewards = []

    
    def compute_td_loss(self, batch_size):
    
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        #Gamma now included in network forward pass, so we need to divide out gamma for current
        #state
        q_values      = self.model(state) / (self.model.gamma)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]

        #Training gamma Model might not work because gamma is not included in forward pass
        #PROBLEM: NN is meant to estimate Q without gamma. Maybe separate NN to estimate gamma?
        #Actor-critic model?
        expected_q_value = reward + next_q_value * (1 - done)
        #self.gamma = self.model.gamma
    
        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #with torch.no_grad():
            #torch.clamp(self.model.gamma, min=0.0, max=1.0)
    
        return loss

    def training_loop(self, num_frames, batch_size, tensorboard = False, writer=None, 
                        run_number = 1, wandb_plot=False, plot_hist = False,
                        hist_vector_gamma = None, hist_vector_rewards = None, 
                        hist_vector_losses = None):

        if wandb_plot:
            wandb.init(
                project= "Gamma Cartpole Training",
                name= f"Experiment_{run_number}",
                config={
                    "environment": "CartPole-v0"
                }
            )

        episode_reward = 0
        gamma_record = []
        reward_record = []
        losses_record = []


        state = self.environment.reset()
        for frame_idx in range(1, num_frames + 1):
            
            #Clamping gamma here instead of in optimizer
            #with torch.no_grad():
                #self.model.gamma = self.model.gamma.clamp(min=0.0, max=1.0)
            
            epsilon_instantiate = epsilon_greedy()
            epsilon = epsilon_instantiate.epsilon_by_frame(frame_idx)
            action = self.model.act(state, epsilon)
    
            next_state, reward, done, _ = self.environment.step(torch.tensor([[action]]).item())
            self.replay_buffer.push(state, action, reward, next_state, done)
    
            state = next_state
            episode_reward += reward
    
            if done:
                state = self.environment.reset()
                self.all_rewards.append(episode_reward)

                if wandb_plot:
                    wandb.log({"Epsilon Seed/Rewards": episode_reward})
                    wandb.log({"Epsilon Seed/Gamma": self.model.gamma.data})

                if tensorboard:
                    writer.add_scalar(f'Episode rewards run {run_number}', episode_reward, frame_idx)
                    writer.add_scalar(f'Gamma run {run_number}', self.gamma, frame_idx)

                #Appending average gamma and reward values from last 1000 trials
                if (num_frames + 1) - frame_idx < 1000:
                    gamma_record.append(self.model.gamma.data)
                    reward_record.append(episode_reward)

                episode_reward = 0
        
            if len(self.replay_buffer) > batch_size:
                loss = self.compute_td_loss(batch_size)
                self.losses.append(loss.data)

                if wandb_plot:
                    wandb.log({"Epsilon Seed/Episode Losses": loss.data})
                    
                if tensorboard:
                    writer.add_scalar(f'Episode Losses run {run_number}',loss.data, frame_idx)

                #Appending average loss value from last 1000 trials
                if (num_frames + 1) - frame_idx < 1000:
                    losses_record.append(loss.data)


        
            if frame_idx % 200 == 0:
                plot(frame_idx, self.all_rewards, self.losses)


        #Plotting averages of metrics into a histogram
        if plot_hist:
            gamma_avg = sum(gamma_record)/len(gamma_record)
            rewards_avg = sum(reward_record)/len(reward_record)
            losses_avg = sum(losses_record)/len(losses_record)

            hist_vector_gamma.append(gamma_avg)
            hist_vector_rewards.append(rewards_avg)
            hist_vector_losses.append(losses_avg)

            data_gamma = [[s] for s in hist_vector_gamma]
            data_rewards = [[s] for s in hist_vector_rewards]
            data_losses = [[s] for s in hist_vector_losses]

            table_gamma = wandb.Table(data=data_gamma, columns=["Average"])
            wandb.log({'Histograms/Gamma Averages-lr=0.0015': wandb.plot.histogram(table_gamma,"Average",
                    title="Ending gamma values")})

            table_rewards = wandb.Table(data=data_rewards, columns=["Average"])
            wandb.log({'Histograms/Reward Averages-lr=0.0015': wandb.plot.histogram(table_rewards,"Average",
                    title="Ending reward values")})

            table_losses = wandb.Table(data=data_losses, columns=["Average"])
            wandb.log({'Histograms/Losses Averages-lr=0.0015': wandb.plot.histogram(table_losses,"Average",
                    title="Ending loss values")})



        wandb.finish()
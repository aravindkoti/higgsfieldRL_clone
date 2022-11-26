import math
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import torch
import random

class epsilon_greedy():
    def __init__(self, epsilon_start = 1.0, epsilon_final = 0.01, epsilon_decay = 500):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (
            (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay))
    
    def plot(self, plot_range = 10000):
        plt.plot([self.epsilon_by_frame(i) for i in range(plot_range)])




def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


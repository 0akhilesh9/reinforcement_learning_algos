import random
import numpy as np
import matplotlib
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T

# Class for storing state transitions for replay
class ReplayMemory(object):
    # init method
    def __init__(self, capacity):
        self.capacity = capacity
        # For current and next states
        self.memory_states = np.zeros((capacity, 3*64*64, 2))
        # For action and rewards
        self.memory_action_rewards = np.zeros((capacity, 2))
        # For done values
        self.memory_done = np.zeros((capacity, 1))
        self.position = 0
        self.counter = 0

    # Inserting into the replay buffer
    def push(self, params):
        self.memory_states[self.position, :] = np.hstack((params[0].reshape(-1,1), params[2].reshape(-1,1)))
        self.memory_action_rewards[self.position, :] = np.hstack((params[1].reshape(-1), params[3]))
        self.memory_done[self.position, :] = params[4]
        self.position = (self.position + 1) % self.capacity
        self.counter = self.counter + 1

    # Randomly sample from the buffer
    def sample(self, batch_size):
        indexes = [random.randrange(0, batch_size) for p in range(0, batch_size)]
        return np.take(self.memory_states,indexes, axis=0), np.take(self.memory_action_rewards, indexes, axis=0), np.take(self.memory_done, indexes, axis=0)

    def __len__(self):
        return self.counter

# Class for DQN
class DQN(nn.Module):
    # Init method
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.lin_layers = nn.Sequential(
            nn.Linear(6*6*64, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 32),
            nn.Linear(32, outputs)
        )

    # Forward propagation
    def forward(self, x):
        x = self.conv(x)
        return self.lin_layers(x.view(x.size(0), -1))
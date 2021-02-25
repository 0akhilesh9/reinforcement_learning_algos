import torch
import torch.nn as nn

# Class for state transition model
class stateTransModel(nn.Module):
    # Init method
    def __init__(self, n_action):
        super(stateTransModel, self).__init__()
        # Action representation layer 1
        self.actionrepr1 = nn.Sequential(
            nn.Linear(n_action, 64),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        # Action representation layer 2
        self.actionconv = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=(3,3), stride=1, padding=1),
            nn.Conv2d(16,32,kernel_size=(1,1),stride=1),
            nn.LeakyReLU(),
        )
        # Regressor module
        self.regressor = nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(32, 32, kernel_size=(1,1), stride=1),
        nn.LeakyReLU(),
        nn.Conv2d(32, 16, kernel_size=(3,3), stride=1, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(16, 32, kernel_size=(1,1), stride=1),
        nn.LeakyReLU()
        )
    # forward method
    def forward(self, x, a):
        action_temp = torch.reshape(self.actionrepr1(a), (-1,32,2,2))
        action_temp = self.actionconv(action_temp)
        new_x = torch.cat((x,action_temp), 1)
        return self.regressor(new_x)

# class for Reward/done detection
class rewardConv(nn.Module):
    # init method
    def __init__(self):
        super(rewardConv, self).__init__()
        # Reward CNN layers
        self.reward_layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(2,2), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1,1), stride=2, padding=1)
        )
        # Reward linear layers
        self.reward_layer2 = nn.Sequential(
            nn.Linear(4, 2),
            nn.Sigmoid()
        )
    # Forward method
    def forward(self, x):
        temp_values = self.reward_layer1(x)
        values = self.reward_layer2(temp_values.view(-1,4))
        return values[:,0], values[:,1]
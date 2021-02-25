import gym
import math
import torch
import random
import itertools
import collections
import numpy as np
import torch.nn as nn
from collections import deque

from nn_models import  FullConnectedNN

# Dataspace for Q Fitted
class DataSpace:
    # Init function
    def __init__(self, buffer_max_size):
        self.buffer_max_size = buffer_max_size
        self.buffer = collections.deque(maxlen=buffer_max_size)
        self.buffer = deque(maxlen=buffer_max_size)
    # Add observations
    def add_obs(self, state_obj, action, reward_val, next_state, done_flag):
        data_sample = (state_obj.reshape(-1), action, np.array([reward_val]), next_state.reshape(-1), done_flag)
        self.buffer.append(data_sample)
    # Get the observations from the memory (here the sample_batch_size is total size of the dataspace)
    def get_samples(self, sample_batch_size):
        batch_state_list = []
        batch_action_list = []
        batch_reward_val_list = []
        batch_next_state_list = []
        batch_done_flag_list = []
        # For shuffling
        batch = random.sample(self.buffer, sample_batch_size)

        for data_sample in batch:
            state_obj, action, reward_val, next_state_obj, done_flag = data_sample
            batch_state_list.append(state_obj)
            batch_action_list.append(action)
            batch_reward_val_list.append(reward_val)
            batch_next_state_list.append(next_state_obj)
            batch_done_flag_list.append(done_flag)
        # Return the batch data
        return (batch_state_list, batch_action_list, batch_reward_val_list, batch_next_state_list, batch_done_flag_list)

    def __len__(self):
        return len(self.buffer)
# Fitted Q Algorithm agent
class FittedQAlgoAgent:
    # Init function
    def __init__(self, env, bin_array, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = env
        self.bin_array = bin_array
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.dataspace = DataSpace(buffer_max_size=buffer_size)
        # GPU usage	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For mapping the NN output to the action space
        self.discrete_to_bin_map = {}
        num_bins = len(self.bin_array)-1
        tmp_list = list(range(num_bins))
        self.discretize_flag = not isinstance(env.action_space, gym.spaces.discrete.Discrete)
        if self.discretize_flag:
            num_actions = env.action_space.sample().shape[0]
        else:
            num_actions = env.action_space.n
        tmp1_list = [[*x] for x in list(itertools.product(*[tmp_list for x in range(num_actions)]))]
        for i in range(len(tmp1_list)):
            self.discrete_to_bin_map[i] = tmp1_list[i]
        
        # Discrete action_space
        if isinstance(env.action_space.sample(),int):
            # Q network
            self.model = FullConnectedNN(env.observation_space.shape, env.action_space.n).to(self.device)
        # Analog action space
        else:
            # Q network
            self.model = FullConnectedNN(env.observation_space.sample().shape, int(math.pow(len(bin_array)-1,env.action_space.sample().shape[0]))).to(self.device)
        # Switch Q network to train mode
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        # Loss function
        self.loss_fn = nn.MSELoss()

    # Get action from the Q network
    def get_action(self, state_obj, discretize=False, epsilon_val=0.05):
        # Convert to tensor
        state_obj = torch.FloatTensor(state_obj.reshape(-1)).float().unsqueeze(0).to(self.device)
        # Feed the state to Q network
        q_logits = self.model.forward(state_obj)
        # Argmax for action
        action_index = np.argmax(q_logits.cpu().detach().numpy())
        
        # Epsilon greedy
        if(np.random.randn() < epsilon_val):
            tmp_action = self.env.action_space.sample()
            # Continuous action space
            if discretize:
                tmp_action = np.digitize(tmp_action, self.bin_array) - 1
            
            return tmp_action
        # Continuous action space
        if discretize:
            action = np.array(self.discrete_to_bin_map[action_index])
        # Discrete action space
        else:
            action = action_index
        return action

    # Loss calculation
    def calculate_loss(self, batch_tuple):    
        # Unpack batch values
        states, actions, reward_vals, next_states, done_flags = batch_tuple
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        reward_vals = torch.FloatTensor(reward_vals).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # Q value from Q network
        current_qvalue = self.model.forward(states).gather(1, actions.unsqueeze(1))
        current_qvalue = current_qvalue.squeeze(1)
        next_qvalue = self.model.forward(next_states)
        
        # Argmax
        max_next_qvalue = torch.max(next_qvalue, 1)[0]
        expected_qvalue = reward_vals.squeeze(1) + self.gamma * max_next_qvalue
        # Loss calculation
        loss_val = 0.5 * self.loss_fn(current_qvalue, expected_qvalue)
        return loss_val
    
    # Backpropagation of loss
    def train_model(self, batch_size):
        batch_data = self.dataspace.get_samples(batch_size)
        loss_val = self.calculate_loss(batch_data)

        self.optimizer.zero_grad()
        # Backpropagation of loss
        loss_val.backward()
        self.optimizer.step()
    # Switch model to eval mode
    def switch_model(self):
        self.model.eval()
import gym
import math
import torch
import random
import itertools
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.autograd import Variable

def to_tensor(x): return torch.from_numpy(x).float()

# Actor model
class ActorModel(nn.Module):
    """
    The model predicts mean of the action distribution space which is used to sample an action. We construct normal
    distribution using this mean and we sample an action from that space.
    """
    # Init method
    def __init__(self, inp_dim, num_actions):
        super().__init__()
        self.n_actions = num_actions
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
            # nn.Linear(32, num_actions),
            # nn.Tanh()
        )
        # mean
        self.linear = nn.Linear(128, self.n_actions)
        # variance
        self.linear_ = nn.Linear(128, self.n_actions)
        # nn.init.kaiming_normal_(self.model[0].weight)
        # nn.init.kaiming_normal_(self.model[2].weight)
        # nn.init.kaiming_normal_(self.model[4].weight)

    # Forward Propagation
    def forward(self, X):
        model_out = self.model(X)
        # # means = model_out[:self.n_actions]
        # means = model_out
        # # variances = model_out[self.n_actions:]

        means = self.linear(model_out)
        variances = self.linear_(model_out)

        return means, variances


# Reinforcec Class
class Reinforce():
    # Init function
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.actor = ActorModel(self.state_dim, self.n_actions)

    # Get action given observation state
    def get_action(self, state, prob = False):
        means, variances = self.actor(to_tensor(state))
        # Sampling an action
        variances = torch_f.softplus(variances)
        eps = torch.randn(means.size())
        action = (means + variances.sqrt() * eps).data

        if prob:
            probability_val = ((-1 * (Variable(action) - means).pow(2) / (2 * variances)).exp()) / ((2 * variances * (Variable(torch.FloatTensor([math.pi]))).expand_as(variances)).sqrt())
            entropy_val = -0.5 * ((variances + 2 * (Variable(torch.FloatTensor([math.pi]))).expand_as(variances)).log() + 1)
            log_probability = probability_val.log()
            return action, log_probability, entropy_val
        return action

    # Train method
    def train(self, max_episodes, gamma=0.99, max_steps=500):
        state_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        # Optimizers for actor and critic
        adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        # To store the experience
        memory = []
        episode_rewards = []
        total_steps = 0

        # Iterating over episodes
        for i in range(max_episodes):
            done = False
            total_reward = 0
            state = self.env.reset().reshape(-1)
            steps = 0
            # Each episode
            while not done:
                # Get action
                action, log_probability, entropy = self.get_action(state, True)
                # Apply action and save the state
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.reshape(-1)
                # Storing
                memory.append((log_probability, entropy, reward, done))

                total_reward += reward
                state = next_state
                steps += 1
                total_steps += 1

                # If terminated or exceeded max_steps
                if done or (steps % max_steps == 0):
                    next_value = 0

                    values, logs_probs, entropies = [], [], []
                    td_targets = np.zeros((len(memory), 1))

                    # target values are calculated backward
                    for i, (log_probs, entropy, reward, done) in enumerate(memory[::-1]):
                        logs_probs.insert(0, log_probs)
                        entropies.insert(0,entropy)

                        next_value = reward + gamma * next_value * (1.0 - done)
                        td_targets[len(memory) - 1 - i] = next_value

                    # Actor update

                    actor_loss = ((-torch.stack(logs_probs) * to_tensor(td_targets).detach()).mean(dim=0) - (0.0001 * sum(entropies)/len(entropies))).mean()

                    adam_actor.zero_grad()
                    actor_loss.backward()
                    adam_actor.step()

                    memory.clear()


            # Save total reward
            episode_rewards.append(total_reward)

        self.switch_model()
        return episode_rewards

    # Switch to eval mode
    def switch_model(self):
        self.actor.eval()
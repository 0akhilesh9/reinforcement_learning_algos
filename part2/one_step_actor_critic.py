import gym
import math
import torch
import random
import itertools
import collections
import numpy as np
import torch.nn as nn
from collections import deque


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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Tanh()
        )
        nn.init.kaiming_normal_(self.model[0].weight)
        nn.init.kaiming_normal_(self.model[2].weight)
        nn.init.kaiming_normal_(self.model[4].weight)
        self.stds = torch.nn.Parameter(torch.Tensor([math.log(0.5)] * num_actions))
    # Forward Propagation
    def forward(self, X):
        means = self.model(X)
        dists = torch.distributions.Normal(means, torch.exp(torch.clamp(self.stds, 1e-3, 50)))

        return dists


# Critic model
class CriticModel(nn.Module):
    # Init method
    def __init__(self, inp_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    # Forward Propagation
    def forward(self, X):
        return self.model(X)

# Actor-Critic Class (Batch variant)
class ActorCriticBatch():
    # Init function
    def __init__(self, env, target=False):
        self.state_dim = env.observation_space.shape[0]
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.actor = ActorModel(self.state_dim, self.n_actions)
        self.critic = CriticModel(self.state_dim)
        self.target = target
        if target:
            self.critic_tmp = CriticModel(self.state_dim)

    # Get action given observation state
    def get_action(self, state, prob = False):
        dists = self.actor(to_tensor(state))
        action = dists.sample()

        with torch.no_grad():
            action = dists.loc + dists.scale * to_tensor(np.random.normal(size=action.shape))
        actions_clamped = torch.clamp(action, self.env.action_space.low.min(), self.env.action_space.high.max())

        # For getting log of probability value. Used for loss calculation purpose.
        if prob:
            return actions_clamped.detach().data.numpy(), dists.log_prob(action)
        return actions_clamped.detach().data.numpy()

    # Train method
    def train(self, max_episodes, gamma=0.99, max_steps=500):
        state_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        # Optimizers for actor and critic
        adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        if self.target:
            adam_critic = torch.optim.Adam(self.critic_tmp.parameters(), lr=1e-3)
        else:
            adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
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
                action, log_prob_val = self.get_action(state, True)
                # Apply action and save the state
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.reshape(-1)
                advantage = reward + (1 - done) * gamma * self.critic(to_tensor(next_state)) - self.critic(to_tensor(state))

                total_reward += reward
                state = next_state
                steps += 1
                total_steps += 1

                critic_loss = advantage.pow(2).mean()
                adam_critic.zero_grad()
                critic_loss.backward()
                adam_critic.step()

                actor_loss = (-log_prob_val * advantage.detach()).mean()
                adam_actor.zero_grad()
                actor_loss.backward()
                adam_actor.step()

                if self.target and (total_steps%1000 == 0):
                    self.update_target()

            # Save total reward
            episode_rewards.append(total_reward)

        self.switch_model()
        return episode_rewards

    # Sync target network parameters
    def update_target(self):
        self.critic.load_state_dict(self.critic_tmp.state_dict())

    # Switch to eval mode
    def switch_model(self):
        self.actor.eval()
        self.critic.eval()
import os
import gym
import sys
import math
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import dqn
import dynamicsmodel


# Program initialization
env = gym.make("Breakout-ram-v0").unwrapped
episode_count=150                   # Max training episodes
max_steps=10000                      # Max steps To terminate current episode
epsilon_val = 0.05                  # For epsilon greedy
n_actions = env.action_space.n      # Environment action space dims
batch_size = 50                     # Batch size for DynaQ training
dqn_update_step = 100               # DQN Target net update step frequency
dqn_policy_update_step = 100        # DQN Policy net update step frequency
discount_factor = 0.99              # Discount factor
dqn_simulate_steps = 100            # Simulation steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Plot function
def plot_fig(plot_list, xlabel, ylabel, title, fig_name):
    plt.figure(fig_name)
    # Line plot
    for plot in plot_list:
        plt.plot(plot[0], plot[1], label=plot[2])
    # Set ticks, legends, title, limits
    x_axis = list(set(plot_list[0][0][:-1]))
    y_axis = list(set(plot_list[0][1]))
    # Set different parameters
    plt.xticks(x_axis, rotation=90)
    # plt.yticks(y_axis)
    plt.yticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(ymin=0)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

# Screen capture from the environment
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # plt.figure()
    # plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()

    resize = T.Compose([T.ToPILImage(), T.Resize((64,64), interpolation=Image.CUBIC), T.ToTensor()])
    screen_val = resize(screen).unsqueeze(0)
    return screen_val

# Action selection using DQN
def select_action(state):
    sample = random.random()
    # Epsilon Random selection
    if (np.random.randn() < epsilon_val):
        tmp_action = env.action_space.sample()
        return torch.tensor(tmp_action)
    # From DQN
    with torch.no_grad():
        a=policy_net(torch.tensor(state, device=device, dtype=torch.float32))
        return a.max(1)[1].view(1, 1)

# Simulation or dreaming
def dyna_simulate(learn_simulate_steps):
    # Update target policy network
    if learn_simulate_steps % dqn_update_step:
        target_net.load_state_dict(policy_net.state_dict())

    # One step
    # Random sampling of state
    sample_states, sample_ar, sample_done = memory.sample(1)
    curr_states = sample_states[:,:,0].reshape(1,3,64,64)
    next_states = sample_states[:, :, 1].reshape(1, 3, 64, 64)
    action_data = np.random.randint(0, n_actions-1, curr_states.shape[0])
    action_data = np.reshape(action_data, (action_data.shape[0],1))
    # Get next state and reward from the Dynamics model
    next_state, reward_val, done_val = dynaq(torch.tensor(curr_states, device=device, dtype=torch.float32), torch.tensor(action_data, device=device, dtype=torch.float32))
    # Get action value for the current state
    state_action_values = policy_net(torch.tensor(curr_states, device=device, dtype=torch.float32)).gather(1, torch.tensor(action_data, device=device, dtype=torch.long)).reshape(-1)
    # For next state using target network
    next_state_action_values = target_net(torch.tensor(next_state, device=device, dtype=torch.float32))
    # Loss calculation
    target_q_vals = reward_val + discount_factor * next_state_action_values.max(1)[0] * (1-done_val)
    dqn_loss = dqn_loss_fn(state_action_values, target_q_vals)
    dqn_optimizer.zero_grad()
    dqn_loss.backward()
    dqn_optimizer.step()

# Train DQN
def update_dqn(learn_steps):
    # Update target policy network
    if learn_steps % dqn_update_step:
        target_net.load_state_dict(policy_net.state_dict())

    # Random sampling
    sample_states, sample_ar, sample_done = memory.sample(batch_size)
    # Actions
    action_data = sample_ar[:,0]
    action_data = np.reshape(action_data, (action_data.shape[0], 1))
    # Current state-action values
    state_action_values = policy_net(torch.tensor(sample_states[:,:,0].reshape(-1, 3, 64, 64), device=device, dtype=torch.float32)).gather(1, torch.tensor(action_data, dtype=torch.long, device=device)).reshape(-1)
    # For next state using target network
    next_state_action_values = target_net(torch.tensor(sample_states[:,:,1].reshape(-1, 3, 64, 64), device=device, dtype=torch.float32))
    # Loss calculation
    target_q_vals = torch.tensor(sample_ar[:,1], device=device) + discount_factor*next_state_action_values.max(1)[0] * torch.tensor(1-sample_done, device=device).reshape(-1)
    dqn_loss = dqn_loss_fn(state_action_values, target_q_vals)
    dqn_optimizer.zero_grad()
    dqn_loss.backward()
    dqn_optimizer.step()

# Train Dyna-Q network - offline
def update_dyna():
    # dynaq.train()
    # Random sampling
    sample_states, sample_ar, sample_done = memory.sample(batch_size)
    curr_states = torch.tensor(sample_states[:,:,0].reshape(-1, 3, 64, 64), device=device, dtype=torch.float32)
    action_data = torch.tensor(np.reshape(sample_ar[:, 0], (sample_ar.shape[0], 1)), device=device, dtype=torch.float32)
    reward_data = torch.tensor(sample_ar[:, 1], device=device)
    # Forward pass Dyna-Q model
    state_trans_out, next_pred_img, dec_state_trans_inp, reward_val, done_val, vae_mu, vae_var = dynaq(curr_states, action_data, train_flag=True)
    next_img = torch.tensor((sample_states[:, :, 1].reshape(-1, 3, 64, 64)), device=device)
    # Next state image encoding from VAE
    enc_next_img = dynaq.vae.reparameterize(*dynaq.vae.encoder(next_img.float()))
    # Loss calculation for individual modules
    l1 = dynamicsmodel.reward_loss_function(reward_data, reward_val, torch.tensor(sample_done, device=device), done_val)
    l2 = dynamicsmodel.state_transition_loss(state_trans_out, enc_next_img, dec_state_trans_inp, next_img)
    l3 = dynamicsmodel.vae_loss_function(dec_state_trans_inp, curr_states, vae_mu, vae_var, batch_size)
    # Backpropagation
    dyna_loss = l1+l2+l3
    dyna_optimizer.zero_grad()
    dyna_loss.sum().backward()
    dyna_optimizer.step()
    # dynaq.eval()

# Train Dyna-Q network - online
def update_dyna(sample_states, sample_ar, sample_done):
    # dynaq.train()
    # Random sampling
    # sample_states, sample_ar, sample_done = memory.sample(batch_size)
    curr_states = torch.tensor(sample_states[:,:,0].reshape(-1, 3, 64, 64), device=device, dtype=torch.float32)
    action_data = torch.tensor(np.reshape(sample_ar[:, 0], (sample_ar.shape[0], 1)), device=device, dtype=torch.float32)
    reward_data = torch.tensor(sample_ar[:, 1], device=device)
    # Forward pass Dyna-Q model
    state_trans_out, next_pred_img, dec_state_trans_inp, reward_val, done_val, vae_mu, vae_var = dynaq(curr_states, action_data, train_flag=True)
    next_img = torch.tensor((sample_states[:, :, 1].reshape(-1, 3, 64, 64)), device=device)
    # Next state image encoding from VAE
    enc_next_img = dynaq.vae.reparameterize(*dynaq.vae.encoder(next_img.float()))
    # Loss calculation for individual modules
    l1 = dynamicsmodel.reward_loss_function(reward_data, reward_val, torch.tensor(sample_done, device=device), done_val)
    l2 = dynamicsmodel.state_transition_loss(state_trans_out, enc_next_img, dec_state_trans_inp, next_img)
    l3 = dynamicsmodel.vae_loss_function(dec_state_trans_inp, curr_states, vae_mu, vae_var, batch_size)
    # Backpropagation
    dyna_loss = l1+l2+l3
    dyna_optimizer.zero_grad()
    dyna_loss.sum().backward()
    dyna_optimizer.step()
    # dynaq.eval()

# Train method
def train_offline():
    cumulative_rewards = []
    # Iterate over episodes
    for i_episode in range(episode_count):
        print("Episode: %d"%i_episode)
        total_reward = 0
        # Initialize the environment and state
        env.reset()
        # Get current state image of the environment
        state = get_screen()
        i = 1
        learn_simulate_steps = 0
        while True:
            # Select and perform an action
            action = select_action(state)
            new_state, reward, done, _ = env.step(action.item())
            total_reward = total_reward + reward
            reward = torch.tensor([reward], device=device)

            # Check if ended
            current_screen = get_screen()
            if not done:
                next_state = current_screen
            else:
                cumulative_rewards.append(total_reward)
                break

            # Store the transition in memory
            memory.push([state.cpu(), action.cpu(), next_state.cpu(), reward.cpu(), done])
            state = next_state

            # Update conditions
            if i % batch_size == 0:
                update_dyna()
            if i % dqn_policy_update_step == 0:
                update_dqn(i)
            # Simulation
            for t in range(dqn_simulate_steps):
                learn_simulate_steps = learn_simulate_steps + 1
                dyna_simulate(i)

            # Terminating condition
            if i % max_steps == 0:
                cumulative_rewards.append(total_reward)
                break
            i = i + 1

    return cumulative_rewards

# Train method
def train_online():
    cumulative_rewards = []
    # Iterate over episodes
    for i_episode in range(episode_count):
        print("Episode: %d"%i_episode)
        total_reward = 0
        # Initialize the environment and state
        env.reset()
        # Get current state image of the environment
        state = get_screen()
        i = 1
        learn_simulate_steps = 0
        while True:
            # Select and perform an action
            action = select_action(state)
            new_state, reward, done, _ = env.step(action.item())
            total_reward = total_reward + reward
            reward = torch.tensor([reward], device=device)

            # Check if ended
            current_screen = get_screen()
            if not done:
                next_state = current_screen
            else:
                cumulative_rewards.append(total_reward)
                break

            # Store the transition in memory
            memory.push([state.cpu(), action.cpu(), next_state.cpu(), reward.cpu(), done])
            state = next_state

            current_next_state = np.expand_dims(np.hstack((state.cpu().reshape(-1,1), next_state.cpu().reshape(-1,1))), axis=0)
            action_reward = np.expand_dims(np.hstack((action.cpu().reshape(-1), reward.cpu())), axis=0)
            update_dyna(current_next_state, action_reward, done)

            # Update conditions
            if i % dqn_policy_update_step == 0:
                update_dqn(i)
            # Simulation
            for t in range(dqn_simulate_steps):
                learn_simulate_steps = learn_simulate_steps + 1
                dyna_simulate(i)

            # Terminating condition
            if i % max_steps == 0:
                cumulative_rewards.append(total_reward)
                break
            i = i + 1

    return cumulative_rewards

# Test program
def test_agent(env_var, num_tests):
    episode_reward = 0.0
    # Iterate over tests
    for test in range(num_tests):
        print("Test- " + str(test))
        env_state = env_var.reset()
        env_state = get_screen()
        done_flag = False
        epsilon_val = 0
        while True:
            # Apply action
            action = select_action(env_state)
            next_state_obj, reward, done_flag, info_details = env_var.step(action)

            episode_reward = episode_reward + reward
            # Terminating condition
            if done_flag:
                break
    return episode_reward / num_tests

# Driver code to run the model
# Offline
env.reset()
# Policy and Target networks
policy_net = dqn.DQN(dynamicsmodel.input_dim, dynamicsmodel.input_dim, n_actions).to(device)
target_net = dqn.DQN(dynamicsmodel.input_dim, dynamicsmodel.input_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()
# Replay buffer
memory = dqn.ReplayMemory(10000)
dqn_loss_fn = nn.MSELoss()
dynaq = dynamicsmodel.DynaQ(64, n_actions).to(device)
# dynaq.eval()
# Optimizer
dqn_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
dyna_optimizer = torch.optim.Adam(dynaq.parameters(), lr=0.01)
algo_list = ["Dyna Q-Offline", "Dyna Q-Online"]
torch.nn.utils.clip_grad_norm(policy_net.parameters(), 0.5)
torch.nn.utils.clip_grad_norm(dynaq.parameters(), 0.5)
num_tests = 10

episode_rewards = []
avg_returns = []
# Offline
episode_rewards.append(train_offline())
avg_returns.append([algo_list[0], test_agent(env, num_tests)])


### Online
env.reset()
# Policy and Target networks
policy_net = dqn.DQN(dynamicsmodel.input_dim, dynamicsmodel.input_dim, n_actions).to(device)
target_net = dqn.DQN(dynamicsmodel.input_dim, dynamicsmodel.input_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()
# Replay buffer
memory = dqn.ReplayMemory(10000)
dqn_loss_fn = nn.MSELoss()
dynaq = dynamicsmodel.DynaQ(64, n_actions).to(device)
# Optimizer
dqn_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
dyna_optimizer = torch.optim.Adam(dynaq.parameters(), lr=0.01)
torch.nn.utils.clip_grad_norm(policy_net.parameters(), 0.5)
torch.nn.utils.clip_grad_norm(dynaq.parameters(), 0.5)

# Online
episode_rewards.append(train_online())
avg_returns.append([algo_list[0], test_agent(env, num_tests)])

for avg_return in avg_returns:
    print(avg_return)

x_values = list(range(1,len(episode_rewards[0])+1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q3_Plot', 'Comparative Analysis of Reinforcement Algorithms for ' + "Breakout-ram-v0")
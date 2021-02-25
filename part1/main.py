import gym
import math
import time
import random
import numpy as np
import pybullet
import pybulletgym
import pybullet_data
import matplotlib.pyplot as plt

from dqn_class import DQNAlgoAgent
from vanilla_q import VanillaQAlgoAgent
from fitted_q import FittedQAlgoAgent
from sarsa import SarsaAlgoAgent


#env_var_name = "CartPole-v0"

# Define common variables
max_steps = 500
batch_size = 32
total_episodes = 100
num_tests = 10
k_val = 1000
# For DQN
target_update_rate = 3
# Discretizing bin
bin_array = [-1.0, 0.0, 1.0]
algo_list = ["Fitted Q", "Vanilla Q", "DQN", "Sarsa"]

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
    plt.yticks(y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(ymin=0)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

# Dynamic epsilon value for greedy epsilon
def get_epsilon(eps):
    return max(0.01, min(1, np.random.exponential(size=total_episodes)[eps]))

# Train method for sarsa
def reinforce_learn_sarsa(env_var, algo_class, max_episodes, max_steps):
    # cumulative reward
    episode_rewards = []
    # Check the action space of the environment
    discretize_flag = not isinstance(env_var.action_space, gym.spaces.discrete.Discrete)
    # Iterate over episodes
    for episode in range(max_episodes):
        state_obj = env_var.reset()
        episode_reward = 0
        # iterate over steps
        for step in range(max_steps):
            # Get an action from the algorithm agent class
            action = algo_class.get_action(state_obj, discretize=discretize_flag, epsilon_val=get_epsilon(episode))
            # For analog action space
            if discretize_flag:
                # Convert discrete action to analog action
                analog_action = []
                for bin in action:
                    analog_action.append(np.random.uniform(bin_array[bin],bin_array[bin+1],1))

                analog_action = [x[0] for x in analog_action]
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(analog_action)
                # Get next action from next state object
                next_action = algo_class.get_action(next_state_obj, discretize=discretize_flag, epsilon_val=get_epsilon(episode))
                # Update the model
                algo_class.update_model((state_obj.reshape(-1), next_action[0], [reward], next_state_obj.reshape(-1), done_flag))
            # For discrete action space
            else:
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(action)
                # Get next action from next state object
                next_action = algo_class.get_action(state_obj, discretize=discretize_flag, epsilon_val=get_epsilon(episode))
                # Update the model
                algo_class.update_model((state_obj.reshape(-1), next_action, [reward], next_state_obj.reshape(-1), done_flag))
            episode_reward = episode_reward + reward
            # Terminating condition
            if (done_flag) or (step == max_steps-1):
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state_obj = next_state_obj
        
    # Switch the NN to eval mode
    algo_class.switch_model()
    return episode_rewards

# Train method for vanilla Q
def reinforce_learn_vanilla_q(env_var, algo_class, max_episodes, max_steps):
    # cumulative reward
    episode_rewards = []
    # Check the action space of the environment
    discretize_flag = not isinstance(env_var.action_space, gym.spaces.discrete.Discrete)
    # Iterate over episodes
    for episode in range(max_episodes):
        state_obj = env_var.reset()
        episode_reward = 0
        # iterate over steps
        for step in range(max_steps):
            action = algo_class.get_action(state_obj, discretize=discretize_flag, epsilon_val=get_epsilon(episode))
            # For analog action space
            if discretize_flag:
                analog_action = []
                for bin in action:
                    analog_action.append(np.random.uniform(bin_array[bin],bin_array[bin+1],1))

                analog_action = [x[0] for x in analog_action]
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(analog_action)
                # Update the model
                algo_class.update_model((state_obj.reshape(-1), action[0], [reward], next_state_obj.reshape(-1), done_flag))
            # For discrete action space
            else:
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(action)
                # Update the model
                algo_class.update_model((state_obj.reshape(-1), action, [reward], next_state_obj.reshape(-1), done_flag))
            episode_reward = episode_reward + reward
            # Terminating condition
            if (done_flag) or (step == max_steps-1):
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state_obj = next_state_obj
        
    # Switch the NN to eval mode
    algo_class.switch_model()
    return episode_rewards

# Train method for fitted Q
def reinforce_learn_fitted_q(env_var, algo_class, max_episodes, max_steps, random_flag=False):
    # cumulative reward
    episode_rewards = []
    # Check the action space of the environment
    discretize_flag = not isinstance(env_var.action_space, gym.spaces.discrete.Discrete)
    # Iterate over episodes
    for episode in range(max_episodes):
        state_obj = env_var.reset()
        episode_reward = 0
        # iterate over steps
        for step in range(max_steps):
            if random_flag:
                epsilon_val=1
            else:
                epsilon_val=get_epsilon(episode)
            action = algo_class.get_action(state_obj, discretize=discretize_flag, epsilon_val=epsilon_val)
            # For analog action space
            if discretize_flag:
                analog_action = []
                for bin in action:
                    analog_action.append(np.random.uniform(bin_array[bin],bin_array[bin+1],1))

                analog_action = [x[0] for x in analog_action]
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(analog_action)
                algo_class.dataspace.add_obs(state_obj, action[0], reward, next_state_obj, done_flag)
            # For discrete action space
            else:
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(action)
                algo_class.dataspace.add_obs(state_obj, action, reward, next_state_obj, done_flag)

            episode_reward = episode_reward + reward
            # Update the model
            if len(algo_class.dataspace) > k_val:
                algo_class.train_model(batch_size)
            # Terminating condition
            if (done_flag) or (step == max_steps-1):
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state_obj = next_state_obj
        
    # Switch the NN to eval mode
    algo_class.switch_model()
    return episode_rewards

# Train method for DQN
def reinforce_learn_dqn(env_var, algo_class, max_episodes, max_steps, batch_size):
    # cumulative reward
    episode_rewards = []
    # Check the action space of the environment
    discretize_flag = not isinstance(env_var.action_space, gym.spaces.discrete.Discrete)
    # Iterate over episodes
    for episode in range(max_episodes):
        state_obj = env_var.reset()
        episode_reward = 0
        # iterate over steps
        for step in range(max_steps):
            action = algo_class.get_action(state_obj, discretize=discretize_flag)
            # For analog action space
            if discretize_flag:
                analog_action = []
                for bin in action:
                    analog_action.append(np.random.uniform(bin_array[bin],bin_array[bin+1],1))

                analog_action = [x[0] for x in analog_action]
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(analog_action)
                algo_class.replay_buffer.add_obs(state_obj, action[0], reward, next_state_obj, done_flag)
            # For discrete action space
            else:
                # Apply the action
                next_state_obj, reward, done_flag, info_details = env_var.step(action)
                algo_class.replay_buffer.add_obs(state_obj, action, reward, next_state_obj, done_flag)
            episode_reward = episode_reward + reward
            # Update the model
            if len(algo_class.replay_buffer) > batch_size:
                algo_class.train_model(batch_size)   
            # Terminating condition
            if (done_flag) or (step == max_steps-1):
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state_obj = next_state_obj
        # Update the model
        if episode % target_update_rate == 0:
            algo_class.update_target()

    return episode_rewards

# Test Function
def test_agent(algo_class, env_var, num_tests):
    # Check the action space of the environment
    discretize_flag = not isinstance(env_var.action_space, gym.spaces.discrete.Discrete)
    episode_reward = 0.0
    # Iterate over tests
    for test in range(num_tests):
        print("Test- " + str(test))
        env_state = env_var.reset()
        done_flag = False
        epsilon_val = 0
        while True:
            #env_var.render()
            # Apply action
            action = algo_class.get_action(env_state, discretize=discretize_flag)
            # For analog action space
            if discretize_flag:
                analog_action = []
                for bin in action:
                    analog_action.append(np.random.uniform(bin_array[bin],bin_array[bin+1],1))

                analog_action = [x[0] for x in analog_action]
                next_state_obj, reward, done_flag, info_details = env_var.step(analog_action)
            # For discrete action space
            else:
                next_state_obj, reward, done_flag, info_details = env_var.step(action)

            episode_reward = episode_reward + reward
            # Terminating condition
            if done_flag:
                # if reward > 0:
                #     print("Goal Reached! " + str(reward))
                # else:
                #     print("Goal Not Reached!!!")
                # time.sleep(2)
                break
    return episode_reward/num_tests

# To store results
inverted_episode_rewards = []
inverted_avg_returns = []
halfcheetah_episode_rewards = []
halfcheetah_avg_returns = []
breakout_episode_rewards = []
breakout_avg_returns = []

# Inverted Pendulum
env_var_name = "InvertedPendulumMuJoCoEnv-v0"
env_var = gym.make(env_var_name)


# Fitted Q
fitted_q_algo_class = FittedQAlgoAgent(env_var, bin_array)
inverted_episode_rewards.append(reinforce_learn_fitted_q(env_var, fitted_q_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[0], test_agent(fitted_q_algo_class, env_var, num_tests)])
# Vanilla Q
vanilla_q_algo_class = VanillaQAlgoAgent(env_var, bin_array)
inverted_episode_rewards.append(reinforce_learn_vanilla_q(env_var, vanilla_q_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[1], test_agent(vanilla_q_algo_class, env_var, num_tests)])
# DQN
dqn_algo_class = DQNAlgoAgent(env_var, bin_array)
inverted_episode_rewards.append(reinforce_learn_dqn(env_var, dqn_algo_class, total_episodes, max_steps, batch_size))
inverted_avg_returns.append([algo_list[2], test_agent(dqn_algo_class, env_var, num_tests)])
# Sarsa
sarsa_algo_class = SarsaAlgoAgent(env_var, bin_array)
inverted_episode_rewards.append(reinforce_learn_sarsa(env_var, sarsa_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[3], test_agent(sarsa_algo_class, env_var, num_tests)])

# Half Cheetah

env_var_name = "HalfCheetahMuJoCoEnv-v0"
env_var = gym.make(env_var_name)

# Fitted Q
fitted_q_algo_class = FittedQAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards.append(reinforce_learn_fitted_q(env_var, fitted_q_algo_class, total_episodes, max_steps))
halfcheetah_avg_returns.append([algo_list[0], test_agent(fitted_q_algo_class, env_var, num_tests)])
# Vanilla Q
vanilla_q_algo_class = VanillaQAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards.append(reinforce_learn_vanilla_q(env_var, vanilla_q_algo_class, total_episodes, max_steps))
halfcheetah_avg_returns.append([algo_list[1], test_agent(vanilla_q_algo_class, env_var, num_tests)])
# DQN
dqn_algo_class = DQNAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards.append(reinforce_learn_dqn(env_var, dqn_algo_class, total_episodes, max_steps, batch_size))
halfcheetah_avg_returns.append([algo_list[2], test_agent(dqn_algo_class, env_var, num_tests)])
# Sarsa
sarsa_algo_class = SarsaAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards.append(reinforce_learn_sarsa(env_var, sarsa_algo_class, total_episodes, max_steps))
halfcheetah_avg_returns.append([algo_list[3], test_agent(sarsa_algo_class, env_var, num_tests)])

# Breakout ram

env_var_name = "Breakout-ram-v0"
env_var = gym.make(env_var_name)

# Fitted Q
fitted_q_algo_class = FittedQAlgoAgent(env_var, bin_array)
breakout_episode_rewards.append(reinforce_learn_fitted_q(env_var, fitted_q_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[0], test_agent(fitted_q_algo_class, env_var, num_tests)])
# Vanilla Q
vanilla_q_algo_class = VanillaQAlgoAgent(env_var, bin_array)
breakout_episode_rewards.append(reinforce_learn_vanilla_q(env_var, vanilla_q_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[1], test_agent(vanilla_q_algo_class, env_var, num_tests)])
# DQN
dqn_algo_class = DQNAlgoAgent(env_var, bin_array)
breakout_episode_rewards.append(reinforce_learn_dqn(env_var, dqn_algo_class, total_episodes, max_steps, batch_size))
inverted_avg_returns.append([algo_list[2], test_agent(dqn_algo_class, env_var, num_tests)])
# Sarsa
sarsa_algo_class = SarsaAlgoAgent(env_var, bin_array)
breakout_episode_rewards.append(reinforce_learn_sarsa(env_var, sarsa_algo_class, total_episodes, max_steps))
inverted_avg_returns.append([algo_list[3], test_agent(sarsa_algo_class, env_var, num_tests)])



# Inverted Pendulum
for avg_return in inverted_avg_returns:
    print(avg_return)

x_values = list(range(1,len(inverted_episode_rewards[0])+1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, inverted_episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q2_Plot', 'Comparative Analysis of Reinforcement Algorithms for ' + "InvertedPendulumMuJoCoEnv-v0")

# Half Cheetah
for avg_return in halfcheetah_avg_returns:
    print(avg_return)

x_values = list(range(1,len(halfcheetah_episode_rewards[0])+1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, halfcheetah_episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q2_Plot', 'Comparative Analysis of Reinforcement Algorithms for ' + "HalfCheetahMuJoCoEnv-v0")

# Breakout ram
for avg_return in breakout_avg_returns:
    print(avg_return)

x_values = list(range(1,len(breakout_episode_rewards[0])+1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, breakout_episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q2_Plot', 'Comparative Analysis of Reinforcement Algorithms for ' + "Breakout-ram-v0")

# Random policy vs Epsilon greedy

env_var_name = "HalfCheetahMuJoCoEnv-v0"
env_var = gym.make(env_var_name)
# Halfcheetah analysis results
halfcheetah_episode_rewards_random=[]
halfcheetah_episode_rewards_epsilon_greedy=[]
halfcheetah_episode_analysis_avg_returns=[]
# Fitted Q random policy
fitted_q_algo_class = FittedQAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards_random.append(reinforce_learn_fitted_q(env_var, fitted_q_algo_class, total_episodes, max_steps, random_flag=True))
halfcheetah_episode_analysis_avg_returns.append(["Random Policy", test_agent(fitted_q_algo_class, env_var, num_tests)])
# Fitted Q epsilon greedy
fitted_q_algo_class = FittedQAlgoAgent(env_var, bin_array)
halfcheetah_episode_rewards_epsilon_greedy.append(reinforce_learn_fitted_q(env_var, fitted_q_algo_class, total_episodes, max_steps))
halfcheetah_episode_analysis_avg_returns.append(["Epsilon greedy", test_agent(fitted_q_algo_class, env_var, num_tests)])

# Print average return
for avg_return in halfcheetah_episode_analysis_avg_returns:    
    print(avg_return)

# Plotting
x_values = list(range(1,len(breakout_episode_rewards[0])+1))
plot_list = []

plot_list.append([x_values, breakout_episode_rewards[0], "Random Policy"])
plot_list.append([x_values, breakout_episode_rewards[1], "Epsilon Greedy"])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q2_Plot', 'Comparative Analysis of Reinforcement Algorithms for ' + env_var_name)
import gym
import math
import time
import random
import numpy as np
import pybullet
import pybulletgym
import pybullet_data
import matplotlib.pyplot as plt

import episodic_actor_critic as ac
import reinforce as rn
import episodic_actor_critic_variant1 as acv
import one_step_actor_critic as aco


# Define common variables
max_episodes = 5
max_steps = 500
num_tests = 10
algo_list = ["Reinforce", "Actor-Critic", "Actor-Critic-variant1", "Actor-Critic-One-Step"]


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
    # plt.xticks(x_axis, rotation=90)
    # plt.yticks(y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(ymin=0)
    plt.title(title)
    plt.legend(loc=2)
    plt.show()

# Dynamic epsilon value for greedy epsilon
def get_epsilon(eps):
    return max(0.01, min(1, np.random.exponential(size=max_episodes)[eps]))

# Test Function
def test_agent(algo_class, env_var, num_tests):
    episode_reward = 0.0
    # Iterate over tests
    for test in range(num_tests):
        print("Test- " + str(test))
        env_state = env_var.reset()
        done_flag = False
        epsilon_val = 0
        while True:
            # Apply action
            action = algo_class.get_action(env_state.reshape(-1))
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
    return episode_reward / num_tests


# To store results
inverted_episode_rewards = []
inverted_avg_returns = []
halfcheetah_episode_rewards = []
halfcheetah_avg_returns = []

# Inverted Pendulum
env_var_name = "InvertedPendulumMuJoCoEnv-v0"
env_var = gym.make(env_var_name)

# Reinforce agent
reinforce_agent = rn.Reinforce(env_var)
inverted_episode_rewards.append(reinforce_agent.train(max_episodes))
inverted_avg_returns.append([algo_list[0], test_agent(reinforce_agent, env_var, num_tests)])
# Actor-Critic agent
actor_critic = ac.ActorCriticBatch(env_var)
inverted_episode_rewards.append(actor_critic.train(max_episodes))
inverted_avg_returns.append([algo_list[1], test_agent(actor_critic, env_var, num_tests)])
# Actor-Critic agent
actor_critic = acv.ActorCriticBatch(env_var)
inverted_episode_rewards.append(actor_critic.train(max_episodes))
inverted_avg_returns.append([algo_list[2], test_agent(actor_critic, env_var, num_tests)])
# Actor-Critic agent
actor_critic = aco.ActorCriticBatch(env_var)
inverted_episode_rewards.append(actor_critic.train(max_episodes))
inverted_avg_returns.append([algo_list[3], test_agent(actor_critic, env_var, num_tests)])

# Half Cheetah
env_var_name = "HalfCheetahMuJoCoEnv-v0"
env_var = gym.make(env_var_name)

# Reinforce agent
reinforce_agent = rn.Reinforce(env_var)
halfcheetah_episode_rewards.append(reinforce_agent.train(max_episodes))
halfcheetah_avg_returns.append([algo_list[0], test_agent(reinforce_agent, env_var, num_tests)])
# Actor-Critic agent
actor_critic = ac.ActorCriticBatch(env_var)
halfcheetah_episode_rewards.append(actor_critic.train(max_episodes))
halfcheetah_avg_returns.append([algo_list[1], test_agent(actor_critic, env_var, num_tests)])
# Actor-Critic agent
actor_critic = acv.ActorCriticBatch(env_var)
inverted_episode_rewards.append(actor_critic.train(max_episodes))
inverted_avg_returns.append([algo_list[2], test_agent(actor_critic, env_var, num_tests)])
# Actor-Critic agent
actor_critic = aco.ActorCriticBatch(env_var)
inverted_episode_rewards.append(actor_critic.train(max_episodes))
inverted_avg_returns.append([algo_list[3], test_agent(actor_critic, env_var, num_tests)])

# Inverted Pendulum
for avg_return in inverted_avg_returns:
    print(avg_return)

x_values = list(range(1, len(inverted_episode_rewards[0]) + 1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, inverted_episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Plot',
         'Comparative Analysis of Reinforcement Algorithms for ' + "InvertedPendulumMuJoCoEnv-v0")

# Half Cheetah
for avg_return in halfcheetah_avg_returns:
    print(avg_return)

x_values = list(range(1, len(halfcheetah_episode_rewards[0]) + 1))
plot_list = []
for i in range(len(algo_list)):
    plot_list.append([x_values, halfcheetah_episode_rewards[i], algo_list[i]])
plot_fig(plot_list, 'Episodes', 'Cumulative Reward', 'Q2_Plot',
         'Comparative Analysis of Reinforcement Algorithms for ' + "HalfCheetahMuJoCoEnv-v0")

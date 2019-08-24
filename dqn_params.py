# this file contains experiment parameters for simple dqn
# set model parameters
import numpy as np


n_env = 16 # size of the grid

episodes = 600
steps = 32

gamma = 0.9
epsilon = np.linspace(0.3, 0.05, episodes)
alpha = 0.0001   # learning rate

n_observations = 1207   # size of input to the network
n_actions = 3           # left, right and forward
n_neurons = 128         # number of units in hidden layer TODO: use tuple for
                        # different hidden layer


# the first reward is top left and then clockwise direction
reward_poses = np.array([[1.5, 1.5], [1.5, 14.5], [14.5, 1.5], [14.5, 14.5]])
reward_target_found, reward_obstacle, reward_free = 10.0, -5.0, 0.0

import numpy as np
import matplotlib.pyplot as plt
import pickle


def running_average(x, window_size, mode='valid'):
	return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


res_folder = 'dqn_main_results/'

reward_of_episodes = pickle.load(open(res_folder + 'reward_of_episodes.pkl', 'rb'))
step_of_episodes = pickle.load(open(res_folder + 'step_of_episodes.pkl', 'rb'))
target_scores = pickle.load(open(res_folder + 'target_scores.pkl', 'rb'))

w = len(scores_manhattan) / 5

fig = plt.figure()
reward_of_episodes_plot = fig.add_subplot(311)
step_of_episodes_plot = fig.add_subplot(312)
target_scores_plot = fig.add_subplot(313)

reward_of_episodes_avg = running_average(reward_of_episodes, w)
step_of_episodes_avg = running_average(step_of_episodes, w)
target_scores_avg = running_average(target_scores, w)

reward_of_episodes_plot.plot(np.arange(len(reward_of_episodes_avg)), reward_of_episodes_avg, linewidth=2)
reward_of_episodes_plot.set_xlabel('Episode')
reward_of_episodes_plot.set_ylabel('Reward')

step_of_episodes_plot.plot(np.arange(len(step_of_episodes_avg)), step_of_episodes_avg, linewidth=2)
step_of_episodes_plot.set_xlabel('Episode')
step_of_episodes_plot.set_ylabel('Step')

target_scores_plot.plot(np.arange(len(target_scores_avg)), target_scores_avg, linewidth=2)
target_scores_plot.set_xlabel('Episode')
target_scores_plot.set_ylabel('Score')

plt.show()

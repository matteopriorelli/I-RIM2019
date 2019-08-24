from __future__ import print_function
import numpy as np
#import logging as lg
import dqn_helpers as dhl
import nrp_helpers as nhl
import time
import rospy
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
from agent import Agent
import dqn_params as param
from termcolor import colored
import tensorflow as tf
import pickle

def run_episode(agent, episode, loss_of_episodes, reward_of_episodes,
                step_of_episodes, target_scores, reward_visits):
    step, episode_done, rewards = 0, 0, []

    # state contains the observation received from the enviornment
    # contructed by the sensory input of the robot
    current_pos, state = dhl.get_observation(nhl.raw_data)

    while not episode_done and step < param.steps:
        # form an observation vector based on the sensors
        prediction_q_actions = agent.feedforward(state)

        # choose action based on e-greedy policy
        if np.random.rand() < param.epsilon[episode]:
            action = np.random.randint(0, param.n_actions)
        else:
            action = int(np.argmax(prediction_q_actions))
        print('Action:', nhl.get_action_name(action))

        # the action goes to the transfer function
        rospy.set_param('action', action)

        # execute action
        action_done = 0
        rospy.set_param('action_done', action_done)

        # wait until action is done, TODO: beutify the TFunc
        while action_done == 0:
            action_done = rospy.get_param('action_done')


        # the robot will take an action, now it is in a next state
        next_pos, next_state = dhl.get_observation(nhl.raw_data)
        print('Position:', int(next_pos[0]), int(next_pos[1]))
        print('Direction:', nhl.raw_data['direction'].data)
        print('-' * 10)


        # check whether the agent receieved the reward
        reward, episode_done, reward_ind = nhl.get_reward(current_pos, next_pos, reward_visits)
        rewards.append(reward)

        # update weights and biases
        with tf.GradientTape() as g:
            prediction_q_actions = agent.feedforward(state)
            target_q_vals = agent.feedforward(next_state)

            target = tf.cast(reward + param.gamma * np.max(target_q_vals), tf.float32)
            loss = agent.get_loss(target, prediction_q_actions)
            loss_of_episodes[episode].append(loss)

        trainable_vars = list(agent.weights.values()) + list(agent.biases.values())
        gradients = g.gradient(loss, trainable_vars)
        agent.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # set next state
        state = next_state
        step += 1

    reward_of_episodes.append(sum(rewards))
    step_of_episodes.append(step)
    target_scores.append(episode_done)


def main():
    exp_desc = 'A simple dqn implementation with 3 layer network.'
    res_folder = 'dqn_main_results/'
    #logger.info("Experiment description: " + exp_desc)
    dhl.tic()

    # subscribe to topics published by ros
    nhl.perform_subscribers()

    # initialize agent
    agent = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)

    loss_of_episodes = [[] for _ in range(param.episodes)]
    reward_of_episodes = []
    step_of_episodes = []
    target_scores = [] # 1 if the agent reached the goal, otherwise 0
    reward_visits = np.zeros((param.n_env, param.n_env), dtype=np.int32)

    for episode in range(param.episodes):
        # time-stamp for the video streaming
        rospy.set_param('i', episode)
        print("----------------- Episode: ", episode)
        # if you want to run the experiment from an external program, use VC
        # this will allow you to use frontend interface from python
        vc = VirtualCoach(environment='local', storage_username='nrpuser',
                          storage_password='password')

        sim = vc.launch_experiment('template_husky_0_0_0_0')
        time.sleep(5)

        # start the experiment
        sim.start()
        nhl.sync_params()

        # inner-loop for running an episode
        run_episode(agent, episode, loss_of_episodes, reward_of_episodes,
                    step_of_episodes, target_scores, reward_visits)

        # stop experiment
        sim.stop()
        time.sleep(10)
        #logger.info("Experiment end: %s", hl.toc())

    # save metrics for postprocessing
    pickle.dump(loss_of_episodes, open(res_folder + 'loss_of_episodes.pkl', 'wb'))
    pickle.dump(reward_of_episodes, open(res_folder + 'reward_of_episodes.pkl', 'wb'))
    pickle.dump(step_of_episodes, open(res_folder + 'step_of_episodes.pkl', 'wb'))
    pickle.dump(target_scores, open(res_folder + 'target_scores.pkl', 'wb'))
    pickle.dump(reward_visits, open(res_folder + 'reward_visits.pkl', 'wb'))

    # save the network
    pickle.dump(agent.weights, open(res_folder + 'weights.pkl', 'wb'))
    pickle.dump(agent.biases, open(res_folder + 'biases.pkl', 'wb'))

    print("Target score: ", sum(target_scores) / param.episodes * 1.0)
    print('Reward visits:', np.sum(reward_visits))


if __name__ == '__main__':
    main()

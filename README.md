# I-RIM2019

This repository contains the code to reproduce the results obtained for the IRIM2019 Conference.

> **Abstract**: This paper presents a spatial navigation task on a mobile robot by employing a deep reinforcement learning algorithm (DQN) in the Neurorobotics Platform (NRP). The navigation task carried out with a Husky robot equipped with a camera and laser sensors. The robot employs sensory readings and position information to select an action and detect a target. The preliminary results show that the NRP can be considered as a realistic virtual environment for robotic agents to perform a navigation task.

## Brief description of the repository folders
**dqn_main_results**: This folder contains the results obtained from the experiment and the weights and biases of the network after the training.
**template_husky_0_0_0_0**: This folder contains the model of the maze, of the Husky robot and the transfer function used to make the robot be able to navigate the environment.
**plots**: This folder contains some plots for the visualization on the video stream, which are updated at every step (for example, the current position of the robot, the score and the loss function).

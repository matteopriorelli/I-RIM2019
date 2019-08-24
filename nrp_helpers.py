import numpy as np
import rospy
import time
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import dqn_params as param

raw_data = {'camera': None, 'laser': None, 'position': None, 'direction': None}

# for the below callbacks you do not need to put in to the nrp tran. funcs.
# get data from rostopic
def get_data(data, args):
    args[0][args[1]] = data


def detect_red():
    """
    Adapted from HBP_CLE tf_*.py
    """
    red_left = red_right = green_blue = 0
    if not isinstance(raw_data['camera'], type(None)):
        lower_red = np.array([0, 30, 30])
        upper_red = np.array([0, 255, 255])
        cv_image = CvBridge().imgmsg_to_cv2(raw_data['camera'], "rgb8")
        # Transform image to HSV (easier to detect colors).
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        # Create a mask where every non red pixel will be a Zero.
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        image_size = (cv_image.shape[0] * cv_image.shape[1])
        if (image_size > 0):
            half = cv_image.shape[1] // 2
            # Get the number of red pixels in the image.
            red_left = cv2.countNonZero(mask[:, :half])
            red_right = cv2.countNonZero(mask[:, half:])
            green_blue = (image_size - (red_left + red_right)) / image_size
            # We have to mutiply the rate by two since it is for an half image only.
            red_left = 2 * (red_left / float(image_size))
            red_right = 2 * (red_right / float(image_size))

#    print(red_left, red_right, type(red_left))
    if red_left > .1 or red_right > .1:
        return True
    else:
        return False


def perform_subscribers():
    rospy.Subscriber('husky/husky/camera', Image, get_data,
                     callback_args=[raw_data, 'camera'])
    rospy.Subscriber('/husky/husky/laser/scan', LaserScan, get_data,
                     callback_args=[raw_data, 'laser'])
    rospy.Subscriber('/gazebo/model_states', ModelStates, get_data,
                     callback_args=[raw_data, 'position'])
    rospy.Subscriber('direction', Int32, get_data,
                     callback_args=[raw_data, 'direction'])

def sync_params():
    rospy.set_param('action_done', 0)
    rospy.set_param('action', -1)
    rospy.set_param('i', 0)

    while any(raw_data[i] is None for i in ['camera', 'laser', 'position', 'direction']):
        time.sleep(0.1)

    time.sleep(10)


def get_action_name(action_id):
    return 'Move forward' if action_id == 0 else 'Turn left' \
           if action_id == 1 else 'Turn right' if action_id == 2 else ''


def get_reward(pos, pos_new, reward_visits):
    if np.linalg.norm(pos - pos_new) < 0.1: # it is safety factor
        return param.reward_obstacle, 0, None
    elif detect_red():
        print('red')
        for r_idx, r_pos in enumerate(param.reward_poses):
            if np.linalg.norm(pos_new - r_pos) <= 0.5:
                reward_visits[int(r_pos[0]), int(r_pos[1])] += 1
                return param.reward_target_found, 1, r_idx
    return param.reward_free, 0, None

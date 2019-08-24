import time
import cv2
import numpy as np
from cv_bridge import CvBridge


def tic():
    """ Log starting time to run specific code block."""

    global _start_time
    _start_time = time.time()


def toc():
    """ Print logged time in hour : min : sec format. """

    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour,t_min) = divmod(t_min, 60)

    return '%f hours: %f mins: %f secs' % (t_hour, t_min, t_sec)


def process_camera(camera):
    # crop the ROI for removing husky front part
    camera = CvBridge().imgmsg_to_cv2(camera, 'bgr8')[:170, :]
    camera = cv2.resize(camera, (32, 32))
    camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    camera = camera.flatten().astype(np.float32)

    return camera


def process_laser(laser):
    laser = np.asarray(laser.ranges)
    laser[laser == np.inf] = 0.0
    laser = laser[np.arange(0, len(laser), 2)]

    return laser


def process_position(position):
    position = position.pose[position.name.index('husky')]
    position = np.asarray([position.position.x, position.position.y])

    return position


def construct_input(position, direction, camera, laser):
    return np.concatenate((position, [direction], camera, laser))


def normalize_input(input):
    minp, maxinp = np.min(input), np.max(input)
    normalized_inp = (input - minp) / (maxinp - minp)

    return normalized_inp[np.newaxis, :]


def get_observation(raw_data):
    position = process_position(raw_data['position'])
    direction = float(raw_data['direction'].data)
    camera, laser = process_camera(raw_data['camera']), \
                    process_laser(raw_data['laser'])
    input = construct_input(position, direction, camera, laser)

    return position, normalize_input(input)

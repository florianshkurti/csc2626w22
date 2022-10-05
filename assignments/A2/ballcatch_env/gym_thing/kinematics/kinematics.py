"""
Author: Ke Dong (kedong0810@gmail.com)
Date: 2020-04-02
Brief: forward and inverse kinematics functions
"""

import math
import json
import numpy as np
import gym
from math import sin, cos
import pyquaternion
from gym.envs.robotics import rotations

FLAG_CALIBRATION = False # you should set FLAG_CALIBRATION to be False

# UR10 D-H parameters
UR10_D = np.array([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])  # unit: m
UR10_A = np.array([0, -0.612, -0.5723, 0, 0, 0])  # unit: m
UR10_ALPHA = np.array([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # unit: radian

# Auxiliary transformations on the real robot
OFFSET_K_TCP_Z = 0.42

OFFSET_K_TCP_Z_ROTATION = 1.309786

OFFSET_CALIBRATION = [-0.0048654,	0.00203897,	-0.0411821,
                      0.00546379, -0.00329907, 0.00237177,
                      -0.000330212, 0.00484434, 0.0615196,
                      0.00, 0.00, 0.00,
                      0.00, 0.00, 0.00]
# make sure configurations of body "calibration_matrix_1", "calibration_matrix_2", "thing_tool" match the parameteres here


# Auxiliary helper functions
def wrap2pi(q):
    """Wraps joint value to be with [-pi, pi]

    Args:
        q: a list of joints value

    Returns:
        wraped array
    """
    return (np.array(q) + np.pi) % (2 * np.pi) - np.pi


def select_closest_solution(q_sols, q_d):
    """Selects the optimal inverse kinemaitcs solutions among a set of feasible joint value
       solutions.
    Args:
        q_sols: A list of a list of feasible joint value solutions (unit: radian)
        q_d: A list of desired joint value solution (unit: radian)
        w: A list of weight corresponding to robot joints
    Returns:
        A list of optimal joint value solution.
    """

    error = []
    for q in q_sols:
        error.append(sum([(q[i] - q_d[i]) ** 2 for i in range(6)]))

    return q_sols[error.index(min(error))]


def homogeneous_transformation_matrix_ur10(i, theta):
    """Calculate the homogeneous transformation matrix (HTM) for the i-th ur10 link
    Args:
        i: an index of joint value.
        theta: A list of joint value solution. (unit: radian)
    Returns:
        An HTM of Link i+1 w.r.t. Link i
    """

    rot_z = np.identity(4)
    rot_z[0, 0] = rot_z[1, 1] = math.cos(theta[i])
    rot_z[0, 1] = -math.sin(theta[i])
    rot_z[1, 0] = math.sin(theta[i])

    trans_z = np.identity(4)
    trans_z[2, 3] = UR10_D[i]

    trans_x = np.identity(4)
    trans_x[0, 3] = UR10_A[i]

    rot_x = np.identity(4)
    rot_x[1, 1] = rot_x[2, 2] = math.cos(UR10_ALPHA[i])
    rot_x[1, 2] = -math.sin(UR10_ALPHA[i])
    rot_x[2, 1] = math.sin(UR10_ALPHA[i])

    matrix = trans_z.dot(rot_z).dot(trans_x).dot(rot_x)

    return matrix


def calibration_matrix(offsets):
    """ Calculates the extra calibration matrix

    :param offsets: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
    :return: T_cal a 4x4 transformation matrix
    """
    trans_x, trans_y, trans_z, rot_x, rot_y, rot_z = offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5]

    T_cal = [ [cos(rot_z) * cos(rot_y), - sin(rot_z) * cos(rot_x) + cos(rot_z) * sin(rot_y) * sin(rot_x), sin(rot_z) *
               sin(rot_x) + cos(rot_z) * sin(rot_y) * cos(rot_x), trans_x],
              [sin(rot_z) * cos(rot_y), cos(rot_z) * cos(rot_x) + sin(rot_z) * sin(rot_y) * sin(rot_x), -cos(rot_z) *
               sin(rot_x) + cos(rot_z) * sin(rot_y) * cos(rot_x), trans_y],
              [-sin(rot_y), cos(rot_y) * sin(rot_x), cos(rot_y) * cos(rot_x), trans_z],
              [0, 0, 0, 1]]

    return np.array(T_cal)

# Forward kinematics functions


def forward_kinematics_ur10(theta, T_cal, index):
    """Calculates the homogeneous transformation matrix given a list of joint values, and calibration matrix

    Args:
        theta: A list of joint values. (unit: radian)
        T_cal: calibration matrix to be added
        index: location for calibration matrix

    Returns:
        The HTM of end-effector joint w.r.t. ur10 base joint

    """
    htm = np.identity(4)

    for i in range(6):
        htm = htm.dot(homogeneous_transformation_matrix_ur10(i, theta))
        if i == index:
            htm = htm.dot(T_cal)

    return htm


def forward_kinematics_arch2tcp(theta):
    """Calculates the homogeneous transformation matrix given a list of joint values.
    Adds those extra transformation matrix on the real robot

    Args:
        theta: a list of joint values. (unit: radian)

    Returns:
        The HTM of end-effector joint w.r.t. ur10 base joint

    """
    offsets = OFFSET_CALIBRATION  if FLAG_CALIBRATION else [0.0 for _ in range(15)]

    gripper_x_offset = offsets[0]
    gripper_y_offset = offsets[1]
    gripper_z_offset = offsets[2]

    T_cal2 = calibration_matrix(offsets[9:])

    T_a0_2_a5 = forward_kinematics_ur10(theta, T_cal2, 1)
    T_a5_2_tcp = np.array([ [sin(OFFSET_K_TCP_Z_ROTATION), 0, cos(OFFSET_K_TCP_Z_ROTATION), gripper_x_offset],
                   [cos(OFFSET_K_TCP_Z_ROTATION), 0, -sin(OFFSET_K_TCP_Z_ROTATION), gripper_y_offset],
                   [0, 1, 0, OFFSET_K_TCP_Z + gripper_z_offset],
                   [0, 0, 0, 1]])

    T_base2tcp = T_a0_2_a5.dot(T_a5_2_tcp)

    return T_base2tcp


def forward_kinematics_odom2tcp(theta):
    """Calculates the homogeneous transformation matrix given a list of joint values
    from the odometry frame to the tool center point frame

    Args:
        theta: [arm_0, arm_1, ..., arm_5, base_x, base_y, base_rotation]

    Returns:
        htm: a HTM from the odometry frame to the tcp frame
    """
    offsets = OFFSET_CALIBRATION if FLAG_CALIBRATION else [0.0 for _ in range(15)]

    T_odom2rb = np.array([
        [math.cos(theta[8]), -math.sin(theta[8]), 0, theta[6]],
        [math.sin(theta[8]), math.cos(theta[8]), 0, theta[7]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_rb2arch = np.array([
        [1, 0, 0, 0.27 ],
        [0, 1, 0, 0.01 ],
        [0, 0, 1, 0.653],
        [0, 0, 0, 1]
    ])
    T_cal1 = calibration_matrix(offsets[3:9])

    return T_odom2rb.dot(T_rb2arch).dot(T_cal1).dot(forward_kinematics_arch2tcp(theta))



def test_forward_kinematics_cpp(verification_pth):
    """Compare the forward kinematics function result here
    with the result from c++ forward kinematics functions

    Args:
        verification_pth: a json file path containing results

    Returns:
        rmse: rooted mean squared error
    """
    running_loss = []
    with open(verification_pth, 'r') as fid:
        config_database = json.load(fid)

        for i in range(len(config_database)):
            theta = np.array(config_database[i]['config'])
            transform_matrix_vec = np.array(config_database[i]['matrix'])

            transform_matrix_python = forward_kinematics_odom2tcp(theta)[:3, :].reshape([-1])

            running_loss.append(np.sum(np.square(transform_matrix_python - transform_matrix_vec)))

    rmse = np.sqrt(np.mean(running_loss))

    return rmse


def test_forward_kinematics_mujoco(verification_pth):
    """Compare the forward kinematics function result with the result returned directly from mujoco simulation

    :param verification_pth: a json file
    :return: rmse error
    """

    # initialize environment
    env = gym.make("gym_thing:thing-v0")

    loss_pos = np.array([0.0, 0.0, 0.0])
    loss_rot = np.zeros((3, 3))

    with open(verification_pth, 'r') as fid:
        config_database = json.load(fid)

        for i in range(len(config_database)):
            theta_nominal = np.array(config_database[i]['config'])
            env.reset(arm_joints=theta_nominal[:6], base_joints=theta_nominal[6:8])
            theta = env.get_obs()[:8].tolist() + [0]

            T_K = forward_kinematics_odom2tcp(theta)
            pos_K = np.array([T_K[0, 3], T_K[1, 3], T_K[2, 3]])
            rotation_K = T_K[:3, :3]

            pos_mujoco = env.sim.data.get_body_xpos("thing_tool")
            rotation_mujoco = pyquaternion.Quaternion(env.sim.data.get_body_xquat("thing_tool")).rotation_matrix

            loss_pos += np.square(pos_K - pos_mujoco)
            loss_rot += np.square(rotation_K - rotation_mujoco)

    rmse_pos = np.sqrt(loss_pos / len(config_database))
    rmse_rot = np.sqrt(loss_rot / len(config_database))

    return rmse_pos, rmse_rot

def forward_kinematics(theta):
    """Calculates the homogeneous transformation matrix given a list of joint values.
    Adds those extra transformation matrix on the real robot

    Args:
        theta: a list of joint values. (unit: radian)

    Returns:
        The HTM of end-effector joint w.r.t. ur10 base joint

    """
    offsets = OFFSET_CALIBRATION  if FLAG_CALIBRATION else [0.0 for _ in range(15)]
    T_cal2 = calibration_matrix(offsets[9:])
    T_a0_2_a5 = forward_kinematics_ur10(theta, T_cal2, 1)
    return T_a0_2_a5

def cup_position_given_theta(theta):
    # W: world frame
    # T: mujoco body "ur10_arm_tool0"
    # C: mujoco body "cup"
    # B: mujoco body "ur10_arm_base_link"

    # constants
    p_W_B_W = np.array([.27, .01, .653])
    p_T_C_T = np.array([0, 0, 0.1])
    R_T_C = rotations.euler2mat(np.radians([ 90, 15, 0]))

    X = np.array(forward_kinematics(theta))
    R_W_T = X[:3, :3]
    p_B_T_W = X[:3, 3]

    # tool relative to world expressed in world frame
    p_W_T_W = p_W_B_W + p_B_T_W

    p_T_C_W = R_W_T.dot(p_T_C_T)

    p_W_C_W = p_W_T_W + p_T_C_W

    R_W_C = R_W_T.dot(R_T_C)

    return p_W_C_W, R_W_C

if __name__ == "__main__":
    import sys
    verification_pth = sys.argv[1]
    rmse_pos = test_forward_kinematics_cpp(verification_pth)
    print(rmse_pos)






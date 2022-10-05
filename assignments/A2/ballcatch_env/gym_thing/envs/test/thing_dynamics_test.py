"""
Compares simulation tracking performance with real robot tracking performance, and tunes certain parameters
Author: Ke Dong (kedong0810@gmail.com)
Date: 2020-03-19
"""
import sys
import os

import gym
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import gym_thing

NUM_DELAY = 1


def rmse(X):
    return np.sqrt(np.mean(np.square(X), axis=0))


def dynamics_test(desired_joints, actual_joints, render=False, xml_file="robot_with_gripper.xml"):
    """
    Compares simulation tracking performance with real tracking performance for a single rosbag
    :param bagname: the path to a rosbag
    :param render: whether render mujoco environments
    :param xml_file: file path to robot description file
    :return: difference between simulation and real joint trajectories
    """

    # create interpretor
    desired_interp, actual_interp = [], []
    for i in range(8):
        desired_interp.append(interp1d(desired_joints[:, 0], desired_joints[:, 1 + i], kind='linear'))
        actual_interp.append(interp1d(actual_joints[:, 0], actual_joints[:, 1 + i], kind='linear'))

    # initializes gym environment
    init_arm_qpos = actual_joints[0][1:7]
    init_base_qpos = actual_joints[0][7:10]
    env = gym.make('thing-v0',
                   initial_arm_qpos=init_arm_qpos,
                   initial_base_qpos=init_base_qpos,
                   n_substeps=10,
                   model_name=xml_file)

    # data collector
    actions = []
    actual_joints_sim = []
    actual_velocity_sim = []
    actual_joints_real = []
    desired_joints_resampled = []
    time_sim = []

    # time related issues
    sim_time = env.sim_time # current simulation time
    max_sim_time = max(desired_joints[:, 0])
    sim_time_next = sim_time + NUM_DELAY * env.dt
    while sim_time_next < max_sim_time:
        if render:
           env.render()
        # retrieve desired and actual joint states
        current_desired_joint_states = []
        for i in range(8):
            current_desired_joint_states.append(desired_interp[i](sim_time))
        current_joint_states_sim = env.get_obs()[:8]
        current_joint_velocity_sim = env.get_obs()[8:]

        # take samples on actual_joint_states
        current_joint_states_actual = []
        for i in range(8):
            current_joint_states_actual.append(actual_interp[i](sim_time))

        action = np.array([desired_interp[i](sim_time_next) for i in range(8)])

        time_sim.append(sim_time)
        actions.append(action)
        actual_velocity_sim.append(current_joint_velocity_sim)
        actual_joints_sim.append(current_joint_states_sim)
        actual_joints_real.append(current_joint_states_actual)
        desired_joints_resampled.append(current_desired_joint_states)

        env.step(action)

        sim_time = env.sim_time
        sim_time_next = sim_time + NUM_DELAY * env.dt

    actions = np.array(actions)
    time_sim = np.array(time_sim)
    actual_velocity_sim = np.array(actual_velocity_sim)
    actual_joints_sim = np.array(actual_joints_sim)
    actual_joints_real = np.array(actual_joints_real)
    desired_joints_resampled = np.array(desired_joints_resampled)

    return actual_joints_sim, actual_velocity_sim, actual_joints_real, desired_joints_resampled, time_sim, actions


def dynamic_test_dir(dataset_pth, xml_file="robot_with_gripper.xml"):
    dataset = np.load(dataset_pth, allow_pickle=True)
    errors = []
    for idx, entry in enumerate(dataset):
        #print("processing %d-th file" % idx)
        desired_joints, actual_joints = entry['desired_trajectory'], entry['actual_trajectory']
        actual_joints_sim, _, actual_joints_real, _, _, _ = dynamics_test(desired_joints, actual_joints, False, xml_file)
        mean_error = rmse(actual_joints_sim - actual_joints_real)
        errors.append(mean_error)
    errors = np.array(errors)

    return errors


def sim_real_comparison_plot(actual_joints_sim, actual_joints_real, desired_joints_resampled, time_sim):

    mean_error = rmse(actual_joints_sim - actual_joints_real)
    print(mean_error)
    for i in range(8):
        plt.figure()
        plt.plot(time_sim, actual_joints_sim[:, i], 'b-', label="actual sim ")
        plt.plot(time_sim, actual_joints_real[:, i], 'g-', label="actual real")
        plt.plot(time_sim, desired_joints_resampled[:, i], 'r-', label='desired')
        plt.legend()
        plt.grid()
        plt.xlabel('time(s)')
        plt.title("The %d-th joint" % (i))

    plt.show()


if __name__ == "__main__":
    argv = sys.argv
    print(np.mean(dynamic_test_dir(argv[1]), axis=0))

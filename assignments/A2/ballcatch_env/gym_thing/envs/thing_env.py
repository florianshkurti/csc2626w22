#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thing mujoco simulation environment
Author: Ke Dong (kedong0810@gmail.com)
        Shichen Lu
Date: 2020-03-19
"""

import os
import math
import copy
import glfw
import numpy as np

import gym
import mujoco_py

from gym import spaces
from gym.utils import seeding

"""Constant configuration parameters"""
DEFAULT_SIZE = 500 # default size used for rgb images

class ThingEnv(gym.Env):
    """
    Description:
        This gym environment provides a basic API to the Thing Mujoco simulation environment. Thing is a mobile
        manipulator, which consists of a 6 DoF UR10 arm and a 3 DoF Ridgeback mobile base. This gym env accepts
        direct joint angles control (including the arm and base), i.e. position servo. no reward calculation or "done"
        classification is provided.

    Observation:
        Type: Box(16)
        Num            Observation              Min         Max
        0-5    arm joint angles(radian)      - 2 * pi     2 * pi
        6-7    base joint angles(meter)         -4          4
        8-13   arm joint velocity (rad/s)      -inf        inf
        14-15  base joint velocity (m/s)       -inf        inf
    Action:(desired position, i.e. position servo)
        Type: Box(8)
        Num            Observation              Min         Max
        0-5    arm joint delta angles          - pi         pi
        6-7    base joint delta positions       -4          4
    Rewards:
        0
    Starting/Resetting state:
        specified by user
    Episode termination:
        No termination from the environment side.

    Important notes:
        1. The Ridgeback base's rotation angle is set to be zero.
        2. No gravity is considered in the current mujoco simulation environment. Experiments show that adding gravity will
        enlarge the difference between robot simulation dynamics and robot real dynamics
    """
    def __init__(self, initial_arm_qpos=(0.0015,-1.947,-2.12,-0.958,-1.5429,1.57),
                       initial_base_qpos=(0.0, 0.0),
                       n_substeps=10,
                       model_name="robot_with_gripper.xml"):
        """
        Initializes a new ThingEnv
        :param initial_arm_qpos: Array of joint angles for initial configuration of robot arm.
                    [t1, t2, t3, t4, t5, t6], units of radians
        :param initial_base_qpos: Array of joint angles for initial configuration of robot base.
                    [x, y], units of meter; Note: base rotation is set to be zero
        :param n_substeps: Number of substeps the simulation runs on every call to step
        :param model_name: robot description file relative path w.r.t "envs/assets/"
        """

        model_fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_name)
        if not os.path.exists(model_fullpath):
            raise IOError('File {} does not exist'.format(model_fullpath))

        # Constants
        self.arm_joint_names = ["ur10_arm_0_shoulder_pan_joint", "ur10_arm_1_shoulder_lift_joint",
                                "ur10_arm_2_elbow_joint", "ur10_arm_3_wrist_1_joint",
                                "ur10_arm_4_wrist_2_joint", "ur10_arm_5_wrist_3_joint"]
        self.arm_act_names = ["ur10_arm_0_shoulder_pan_act", "ur10_arm_1_shoulder_lift_act", "ur10_arm_2_elbow_act",
                              "ur10_arm_3_wrist_1_act", "ur10_arm_4_wrist_2_act", "ur10_arm_5_wrist_3_act"]
        self.arm_body_names = ["ur10_arm_shoulder_link", "ur10_arm_upper_arm_link", "ur10_arm_forearm_link",
                               "ur10_arm_wrist_1_link", "ur10_arm_wrist_2_link", "ur10_arm_wrist_3_link"]
        self.base_joint_names = ["ridgeback_x", "ridgeback_y"]
        self.base_act_names = ["ridgeback_x_act", "ridgeback_y_act"]
        self.act_names = self.arm_act_names +  self.base_act_names

        # Action consists of delta joint angle command
        act_high = np.array([math.pi for _ in range(6)] + [4 for _ in range(2)], dtype=np.float32)
        self.action_space = spaces.Box(low = -act_high, high = act_high, dtype='float32')

        obs_high = np.array([2*math.pi] * len(self.arm_joint_names) +
                            [4] * len(self.base_joint_names) +
                            [np.finfo(np.float32).max] * ( len(self.arm_joint_names) + len(self.base_joint_names) ),
                            dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype='float32')

        # Initialize mujoco configuration
        self.model = mujoco_py.load_model_from_path(model_fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        # Initialize viewers
        self.viewer = None
        self._viewers = {}

        # Initialize offscreen render context for kinect camera, if depth image is wanted
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
        # self.sim.render(mode='window', camera_name='kinect', width=240, height=240, depth=False)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # Setup our training environment
        self.seed()

        self.initial_arm_qpos = initial_arm_qpos
        self.initial_base_qpos = initial_base_qpos

        self._env_setup(initial_arm_qpos, initial_base_qpos)
        self.sim.forward()
        self.initial_state = copy.deepcopy(self.sim.get_state())

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    @property
    def sim_time(self):
        return self.sim.data.time

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._set_control(action)
        self.sim.step()
        self._step_callback()
        obs = self.get_obs()

        reward = self.reward()

        info = {}

        return obs, reward, 0, info

    def reward(self):
            return 0

    def reset(self, arm_joints=None, base_joints=None):

        self.sim.set_state(self.initial_state)

        # set joint states with static velocity
        if arm_joints is None:
            arm_joints = self.initial_arm_qpos
        if base_joints is None:
            base_joints = self.initial_base_qpos
        self._env_setup(arm_joints, base_joints)

        self.sim.forward()

        obs = self.get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    # Handy helper methods:

    def get_obs(self):
        """
        Return the observation for current sim state.
        Currently, observation consists of: arm state, base state
        Camera information is currently not included
        """
        arm_pos, arm_vel = self._get_arm_joint(self.sim)
        base_pos, base_vel = self._get_base_joint(self.sim)
        obs = np.hstack((arm_pos, base_pos, arm_vel, base_vel))

        # im_rgb, im_depth = self.sim.render(mode='offscreen', camera_name='kinect', width=240, height=240, depth=True)
        # obs['camera_depth'] = im_depth
        # obs['camera_rgb'] = im_rgb
        # obs['ball_pos'] = self.sim.data.get_body_xpos('ball')
        #
        # if self.viewer is not None:
        #     self.viewer.render(240, 240, 0)
        #     im_rgb, im_depth = self.viewer.read_pixels(240, 240)
        #     obs['camera_depth'] = im_depth
        #     obs['camera_rgb'] = im_rgb
        return obs

    def get_ctrl(self):
        """
        Returns the current desired joint anlges
        :return:
        """
        control = []
        for i in range(len(self.act_names)):
            control.append(self.sim.data.ctrl[self.model.actuator_name2id(self.act_names[i])])
        return control

    # Private extension methods
    # ----------------------------

    def _get_arm_joint(self, sim):
        """
        Return a list of current arm joint positions and velocity
        """
        if sim.data.qpos is not None and sim.model.joint_names:
            # names = [n for n in sim.model.joint_names if n.startswith('ur10_arm')]
            return (
                np.array([sim.data.get_joint_qpos(name) for name in self.arm_joint_names]),
                np.array([sim.data.get_joint_qvel(name) for name in self.arm_joint_names]),
            )
        return np.zeros(0), np.zeros(0)

    def _get_base_joint(self, sim):
        """
        Return a list of current base joint positions and velocity
        """
        if sim.data.qpos is not None and sim.model.joint_names:
            return (
                np.array([sim.data.get_joint_qpos(name) for name in self.base_joint_names]),
                np.array([sim.data.get_joint_qvel(name) for name in self.base_joint_names]),
            )
        return np.zeros(0), np.zeros(0)

    def _env_setup(self, initial_arm_qpos, initial_base_qpos):
        """
        Setup joint positions of the interacting objects in the environment, joint velocities are set to be zero
        :param initial_arm_qpos: Array of joint angles for initial configuration of robot arm.
                                [t1, t2, t3, t4, t5, t6], units of radians
        :param initial_base_qpos: Array of joint positions for initial configuration of base position.
                                [p_x, p_y] wrt. global frame. p_z is always 0 for base.
        :return: None
        """
        for i in range(len(self.arm_joint_names)):
            self.sim.data.set_joint_qpos(self.arm_joint_names[i], initial_arm_qpos[i])
        for i in range(len(self.base_joint_names)):
            self.sim.data.set_joint_qpos(self.base_joint_names[i], initial_base_qpos[i])
        ctrl = np.hstack((initial_arm_qpos, initial_base_qpos))
        self._set_control(ctrl)
        self.sim.forward()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _viewer_setup(self):
        """
        Initial configuration of the viewer. Can be used to set the camera position,
        """
        pass

    def _render_callback(self):
        """
        A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """
        A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _set_control(self, control):
        """
        Sets the given control to the simulation. Input of desired joint angle values. (Not displacement)
        """
        for i in range(len(self.act_names)):
            self.sim.data.ctrl[self.model.actuator_name2id(self.act_names[i])] = control[i]

    def _is_success(self, achieved_goal=None, desired_goal=None):
        """
        Determine if we have achieved the goal. Currently checks only if ball is close to cup
        :param achieved_goal: The goal we have currently achieved
        :param desired_goal: The goal we want to achieve
        :return: Boolean. True if we have succeeded in achieving the goal
        """
        # position_distance, rotation_distance = self.goal_distance(achieved_goal, desired_goal)
        #
        # return (position_distance < self.pos_succ_threshold
        #         and
        #         rotation_distance < self.rot_succ_threshold).astype(np.float32)
        pass

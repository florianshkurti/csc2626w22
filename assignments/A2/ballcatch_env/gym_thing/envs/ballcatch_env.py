#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import copy
import glfw
import numpy as np
import scipy.spatial.transform
import scipy.spatial.distance
from pyquaternion import Quaternion
import gym
import mujoco_py
from gym.envs.robotics import rotations
from gym import spaces
from gym.utils import seeding

import gym_thing.kinematics.kinematics as kin
import gym_thing.nlopt_optimization.nlopt_optimizer as nlopt_optimizer
import gym_thing.nlopt_optimization.nlopt_functions as nlopt_functions

DEFAULT_SIZE = 500 # default size used for rgb images

class ThingBallCatchEnv(gym.Env):
    """
    Environment for simulating a ball catching task using a UR10 robotic arm attached to a Clearpath Ridgeback mobile base
    Goal: EE pose vector [x, y, z, qw, qi, qj, qk] for cartesian position and quaternion rotation
    """
    def __init__(self, model_path, nlopt_config_path, initial_arm_qpos, initial_ball_qvel, initial_ball_qpos, initial_base_qpos, n_substeps,
                 pos_rew_weight=1, rot_rew_weight=1, online=True, training_data=None,
                 gravity_factor_std=0., n_substeps_std=0., mass_std=0., act_gain_std=0., joint_noise_std=0., vicon_noise_std=0., control_noise_std=0.,
                 temp_penalty_type=0, random_training_index=True):
        """
        Initializes a new ThingEnv
        :param model_path: Path to mujoco xml file for robot
        :param nlopt_config_path: Path to json file for nlopt config
        :param initial_arm_qpos: Array of joint angles for initial configuration of robot arm. [t1, t2, t3, t4, t5, t6], units of radians
        :param initial_ball_qvel: Array of joint velocities for intial configuration of ball velocity. [v_x, v_y, v_z] wrt. global frame
        :param initial_ball_qpos: Array of joint positions for initial configuration of ball position. [p_x, p_y, p_z] wrt. global frame
        :param n_substeps: Number of substeps the simulation runs on every call to step
        :param pos_rew_weight: Coefficient multiplier for position reward
        :param rot_rew_weight: Coefficient multiplier for rotation reward
        :param online: Set to true to generate ball trajectories and goal positions online. Set to false to read them from training_data
        :param training_data: np.array of ball trajectories and goal positions
        :param gravity_factor: multiplier to gravity accel applied to ball (should be equal to ball weight)
        """

        # Check model path valid, parse whether full path or just name given
        if model_path.startswith('/'):
            model_fullpath = model_path
        else:
            model_fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(model_fullpath):
            raise IOError('File {} does not exist'.format(model_fullpath))

        # Check nlopt config path valid, parse wether full path or just name given
        if nlopt_config_path.startswith('/'):
            nlopt_fullpath = nlopt_config_path
        else:
            nlopt_fullpath = os.path.join(os.path.dirname(__file__), 'nlopt_optimization', 'config', nlopt_config_path)
        if not os.path.exists(nlopt_fullpath):
            raise IOError('File {} does not exist'.format(nlopt_fullpath))

        # Constants
        self.arm_joint_names = ["ur10_arm_0_shoulder_pan_joint", "ur10_arm_1_shoulder_lift_joint", "ur10_arm_2_elbow_joint", "ur10_arm_3_wrist_1_joint",
                                "ur10_arm_4_wrist_2_joint", "ur10_arm_5_wrist_3_joint"]
        self.arm_act_names = ["ur10_arm_0_shoulder_pan_act", "ur10_arm_1_shoulder_lift_act", "ur10_arm_2_elbow_act", "ur10_arm_3_wrist_1_act",
                              "ur10_arm_4_wrist_2_act", "ur10_arm_5_wrist_3_act"]
        self.arm_body_names = ["ur10_arm_shoulder_link", "ur10_arm_upper_arm_link", "ur10_arm_forearm_link", "ur10_arm_wrist_1_link", "ur10_arm_wrist_2_link",
                               "ur10_arm_wrist_3_link"]
        self.ball_joint_names = ["ball_x", "ball_y", "ball_z"]
        self.base_joint_names = ["ridgeback_x", "ridgeback_y"]
        self.base_act_names = ["ridgeback_x_act", "ridgeback_y_act"]

        # Action consists of delta joint angle command
        self.action_space = spaces.Box(low = np.array([-0.02*math.pi]*6, dtype=np.float32),
                                       high = np.array([0.02*math.pi]*6, dtype=np.float32),
                                       dtype='float32')

        # Observation consists of joint angles and velocities of arm and position of ball over last three timesteps
        # self.observation_space = spaces.Box(-2*math.pi, 2*math.pi, shape=(len(self.arm_joint_names)*2,), dtype='float32')
        self.observation_space = spaces.Box(low=np.array([-2*math.pi] * (len(self.arm_joint_names)*2) + [-10] * 9, dtype=np.float32),
                                            high=np.array([2*math.pi] * (len(self.arm_joint_names)*2) + [10] * 9, dtype=np.float32),
                                            dtype='float32')
        self.prev_ball_obs1 = np.zeros((3,))
        self.prev_ball_obs2 = np.zeros((3,))

        # Parameters for goal and reward calculation
        self.pos_rew_weight = pos_rew_weight
        self.rot_rew_weight = rot_rew_weight

        # Initialize mujoco configuration
        self.model = mujoco_py.load_model_from_path(model_fullpath)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.sim.data.xfrc_applied[self.model.body_name2id('ball')][2] = -9.81 * 0.25  # Add gravity to ball

        # Initialize viewers
        self.viewer = None
        self._viewers = {}

        # Initialize offscreen render context for kinect camera
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
        # self.sim.render(mode='window', camera_name='kinect', width=240, height=240, depth=False)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # Setup our training environment
        self.seed()
        self.online = online
        if not self.online and training_data is None:
            raise ValueError("No training data provided for offline training")
        self.training_data = training_data
        self._training_index = -1
        self.random_training_index = random_training_index
            

        self.initial_arm_qpos = initial_arm_qpos
        self.initial_base_qpos = initial_base_qpos
        self.initial_ball_qvel = initial_ball_qvel
        self.initial_ball_qpos = initial_ball_qpos

        self._env_setup(initial_arm_qpos, initial_ball_qvel, initial_ball_qpos, initial_base_qpos)
        self.initial_ctrl = np.hstack((self._get_arm_joint(self.sim)[0], self._get_base_joint(self.sim)[0]))
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.achieved_goal_xyzquat = self._get_achieved_goal_xyzquat()
        self.achieved_goal_joint = self._get_achieved_goal_joint()

        # For dynamics randomization
        self.initial_mass = self.sim.model.body_mass.copy()
        self.initial_act_gain = self.sim.model.actuator_gainprm.copy()
        self.initial_gravity_factor = 0.25
        self.initial_n_substeps = n_substeps

        self.mass_std = mass_std
        self.act_gain_std = act_gain_std
        self.gravity_factor_std = gravity_factor_std
        self.n_substeps_std = n_substeps_std

        self.joint_noise_std = joint_noise_std
        self.vicon_noise_std = vicon_noise_std
        self.control_noise_std = control_noise_std

        # Optimal ball catch location calculation
        self.nlopt_success = None
        self.nlopt_config_path = nlopt_fullpath
        self.nlopt_opt = nlopt_optimizer.NloptOptimizer(self.nlopt_config_path)
        self.goal = np.array(self.get_optimal_catching_pos())[0:6]  # Currently, get_optimal_catching_pos() returns joint states for both arm and base

        # For previous inv kin stuff
        self.action_done_once = False
        self.q_sol = []

        # Temp
        self.temp_penalty_type = temp_penalty_type

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.prev_ball_obs2 = self.prev_ball_obs1
        self.prev_ball_obs1 = self.sim.data.get_body_xpos('ball')
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        # self.achieved_goal = self._get_achieved_goal_xyzquat()
        # self.achieved_goal_joint = self._get_achieved_goal_joint()
        # self.achieved_goal_xyzquat = self._get_achieved_goal_xyzquat()

        done = self._is_success()#self.achieved_goal_xyzquat)
        info = {
            'is_success': done,
            'nlopt_success' : self.nlopt_success,
            'arm_joint_state': self._get_arm_joint(self.sim),
            'arm_joint_acc' : np.linalg.norm(self.sim.data.qacc[5:11])
        }
        # reward = self.compute_reward(self.achieved_goal_joint, self.goal, info)
        reward = self.compute_reward(obs, self.goal, info)
        if done: reward += 10
        # reward = 1 if done else 0
        return obs, reward, done, info

    def reset(self, arm_joints=None, ball_xpos=None, ball_qvel=None):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim(arm_joints=arm_joints, ball_xpos=ball_xpos, ball_qvel=ball_qvel)
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            # self.viewer.close()
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

    # Extension methods
    # ----------------------------

    def _reset_sim(self, arm_joints=None, ball_xpos=None, ball_qvel=None):
        if self.mass_std != 0:
            new_mass = self.initial_mass * np.clip(np.random.normal(1, self.mass_std, self.initial_mass.shape), 0.5, 1.5)
            for i in range(len(self.sim.model.body_mass)):
                self.sim.model.body_mass[i] = new_mass[i]
        if self.act_gain_std != 0:
            new_act_gain = self.initial_act_gain * np.clip(np.random.normal(1, self.act_gain_std, self.initial_act_gain.shape), 0.5, 1.5)
            for i in range(self.sim.model.actuator_gainprm.shape[0]):
                self.sim.model.actuator_gainprm[i] = new_act_gain[i]
        if self.gravity_factor_std != 0:
            new_gravity_factor = self.initial_gravity_factor * np.clip(np.random.normal(1, self.gravity_factor_std), 0.5, 1.5)
            self.sim.data.xfrc_applied[self.model.body_name2id('ball')][2] = -9.81 * new_gravity_factor  # Add gravity to ball
        if self.n_substeps_std != 0:
            new_substeps = self.initial_n_substeps * np.clip(np.random.normal(1, self.n_substeps_std), 0.5, 1.5)
            self.sim.nsubsteps = int(new_substeps)
        self.sim.set_constants()

        self.prev_ball_obs1 = np.zeros((3,))
        self.prev_ball_obs2 = np.zeros((3,))

        self.sim.set_state(self.initial_state)
        for i in range(len(self.initial_ctrl)):
            self.sim.data.ctrl[i] = self.initial_ctrl[i]

        if self.online:
            new_ball_qvel = ball_qvel if ball_qvel is not None else (self.initial_ball_qvel * np.clip(np.random.normal(1, .1, self.initial_ball_qvel.shape), 0, 2))  # Add noise
            new_ball_qpos = ball_xpos if ball_xpos is not None else (self.initial_ball_qpos * np.clip(np.random.normal(1, .1, self.initial_ball_qpos.shape), 0, 2))  # Add noise

            for i in range(len(self.ball_joint_names)):
                self.sim.data.set_joint_qpos(self.ball_joint_names[i], new_ball_qpos[i])
                self.sim.data.set_joint_qvel(self.ball_joint_names[i], new_ball_qvel[i])

            if arm_joints is not None:
                for i in range(len(self.arm_joint_names)):
                    self.sim.data.set_joint_qpos(self.arm_joint_names[i], arm_joints[i])

            self.sim.forward()

            opt_res = self.get_optimal_catching_pos()
            self.goal = np.array(opt_res)[0:6]

            did_reset_sim = self.nlopt_success
        else:
            rnd_data_idx = self.select_training_index()

            selected_data = self.training_data[rnd_data_idx, :]

            new_ball_qpos = selected_data[0:3]
            new_ball_qvel = selected_data[3:6]

            for i in range(len(self.ball_joint_names)):
                self.sim.data.set_joint_qpos(self.ball_joint_names[i], new_ball_qpos[i])
                self.sim.data.set_joint_qvel(self.ball_joint_names[i], new_ball_qvel[i])
            self.sim.forward()

            new_goal = selected_data[6:12]
            self.goal = new_goal

            did_reset_sim = True

        return did_reset_sim

    def select_training_index(self):
        if self.random_training_index:
            index = np.random.randint(0, self.training_data.shape[0])
        else:
            index = (self._training_index + 1) % self.training_data.shape[0]
        self._training_index = index
        return index

    def _viewer_setup(self):
        """
        Initial configuration of the viewer. Can be used to set the camera position,
        for example.
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

    def _set_action(self, action):
        """
        Applies the given action to the simulation. Input of joint angle displacement
        """
        action += np.random.normal(0, self.control_noise_std, action.shape)  # Add noise
        for i in range(len(self.arm_act_names)):
            self.sim.data.ctrl[self.model.actuator_name2id(self.arm_act_names[i])] += action[i]

    # RobotEnv Methods
    def _set_action_inv_kin(self, action):
        """
        Applies the given action to the simulation. Solves for joint movement required using inverse kinematics.
        :param action: Vector consisting of [delta_arm_x, delta_arm_y, delta_arm_z, q.w, q.i, q.j. q.k]
        to specify the way we wish to move the robotic arm as a delta from current position/rotation
        (base movement currently not supported)
        """
        # 1. get current tcp pos
        ur10_theta, _ = self._get_arm_joint(self.sim)
        T_arch2tcp_cur = kin.forward_kinematics_arch2tcp(ur10_theta)

        # 2. get the transformation matrix base on deltaTCP
        # quat = Quaternion(action[3:7])
        # unit_quat = quat.normalised
        rotation_mat = rotations.euler2mat(action[3:6])
        # rotation_mat = unit_quat.rotation_matrix
        trans_arm = [[action[0]], [action[1]], [action[2]]]
        arm_mat = np.hstack((rotation_mat, trans_arm))
        arm_mat = np.vstack((arm_mat, [0, 0, 0, 1]))

        # 3. get desired TCP pose, use IK to get corresponding joint values
        T_arch2tcp_des = T_arch2tcp_cur * arm_mat
        self.q_sol = kin.inverse_kinematics_arch2tcp(T_arch2tcp_des, ur10_theta)
        if len(self.q_sol) == 0:
            return 0#-1

        # 4. send desired joint values to the simulator
        for i in range(len(self.arm_act_names)):
            self.sim.data.ctrl[self.model.actuator_name2id(self.arm_act_names[i])] = self.q_sol[i]

        return 0

    def _get_obs(self):
        """
        Return the observation for current sim state.
        Currently, observation consists of: arm state, base state
        Camera information is currently not included
        """
        arm_pos, arm_vel = self._get_arm_joint(self.sim)
        ball_pos = self.sim.data.get_body_xpos('ball')
        # base_pos, base_vel = self._get_base_joint(self.sim)

        # Add noise
        arm_pos += np.random.normal(0, self.joint_noise_std, arm_pos.shape)
        arm_vel += np.random.normal(0, self.joint_noise_std, arm_vel.shape)
        ball_pos += np.random.normal(0, self.vicon_noise_std, ball_pos.shape)

        obs = np.hstack((arm_pos, arm_vel, ball_pos, self.prev_ball_obs1, self.prev_ball_obs2))


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
            # names = [n for n in sim.model.joint_names if n.startswith('ridgeback')]
            return (
                np.array([sim.data.get_joint_qpos(name) for name in self.base_joint_names]),
                np.array([sim.data.get_joint_qvel(name) for name in self.base_joint_names]),
            )
        return np.zeros(0), np.zeros(0)

    def _set_random_arm_joint_and_ctrl(self):
        rand_pos, rand_quat = self._random_pose_in_sphere(radius=0.8, xpos=0.5, zpos=0.8, zscale=0.8)
        rotation = scipy.spatial.transform.Rotation.from_quat(rand_quat)
        rotpos = np.hstack((rotation.as_matrix(), rand_pos.reshape([3,1])))
        T_mtx = np.vstack((rotpos, np.array([0,0,0,1])))
        arm_qpos_all = kin.inverse_kinematics_arch2tcp(T_mtx)
        selected_qpos = int(self.np_random.uniform(0, 7))

        if len(arm_qpos_all) == 0:
            return False
        else:
            arm_qpos = arm_qpos_all[selected_qpos, :]

            for i in range(len(self.arm_joint_names)):
                self.sim.data.set_joint_qpos(self.arm_joint_names[i], arm_qpos[i])
                self.sim.data.ctrl[self.model.actuator_name2id(self.arm_act_names[i])] = arm_qpos[i]

            return True

    def _env_setup(self, initial_arm_qpos, initial_ball_qvel, initial_ball_qpos, initial_base_qpos):
        """
        Setup joint positions/velocities of the interacting objects in the environment.
        :param initial_arm_qpos: Array of joint angles for initial configuration of robot arm.
                                [t1, t2, t3, t4, t5, t6], units of radians
        :param initial_ball_qvel: Array of joint velocities for intial configuration of ball velocity.
                                [v_x, v_y, v_z] wrt. global frame
        :param initial_ball_qpos: Array of joint positions for initial configuration of ball position.
                                [p_x, p_y, p_z] wrt. global frame
        :param initial_base_qpos: Array of joint positions for initial configuration of base position.
                                [p_x, p_y] wrt. global frame. p_z is always 0 for base.
        :return: None
        """
        for i in range(len(self.arm_joint_names)):
            self.sim.data.set_joint_qpos(self.arm_joint_names[i], initial_arm_qpos[i])
        for i in range(len(self.ball_joint_names)):
            self.sim.data.set_joint_qpos(self.ball_joint_names[i], initial_ball_qpos[i])
        for i in range(len(self.ball_joint_names)):
            self.sim.data.set_joint_qvel(self.ball_joint_names[i], initial_ball_qvel[i])
        for i in range(len(self.base_joint_names)):
            self.sim.data.set_joint_qpos(self.base_joint_names[i], initial_base_qpos[i])
        self.sim.forward()
        self.initial_ee_xpos = self.sim.data.get_body_xpos('thing_tool')

    def _sample_goal(self):
        """
        Sample a goal from our goal space
        :return: Copy of the sampled goal
        """
        # pos_goal = self.initial_ee_xpos + self.np_random.uniform(-0.15, 0.15, size=self.initial_ee_xpos.shape)
        # rot_goal = self.sim.data.get_body_xquat("ur10_arm_ee_link")

        pos_goal, rot_goal = self._random_pose_in_sphere(radius=0.8, xpos=0.5, zpos=0.8, zscale=0.8)

        goal = np.hstack((pos_goal, rot_goal))
        return goal.copy()

    def _get_achieved_goal_xyzquat(self):
        '''
        Get current state of "cup" body
        :return: Numpy Array (1x7) of [cup_xpos, cup_xquat]
        '''
        return np.hstack((self.sim.data.get_body_xpos("cup"), self.sim.data.get_body_xquat("cup")))

    def _get_achieved_goal_joint(self):
        res = np.zeros((6,))
        for i in range(6):
            res[i] = self.sim.data.get_joint_qpos(self.arm_joint_names[i])
        return res

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

        # # check if ball within sphere around cup body origin
        # position_distance = np.linalg.norm(achieved_goal[0:3] - self.sim.data.get_body_xpos('ball'))
        # return position_distance < self.pos_succ_threshold

        # check if ball within cylinder defined by axis from cup body origin to
        # ur10_arm_ee_link body origin
        # return self.is_point_in_cylinder(self.sim.data.get_body_xpos("cup"),
        #                                  self.sim.data.get_body_xpos("ur10_arm_ee_link"),
        #                                  0.01,
        #                                  0.0025,
        #                                  self.sim.data.get_body_xpos('ball'))

        return self.is_point_in_cylinder(self.sim.data.get_body_xpos("cup"),
                                         self.sim.data.get_body_xpos("ur10_arm_ee_link"),
                                         0.2,
                                         0.05,
                                         self.sim.data.get_body_xpos('ball'))

    def _random_pose_in_sphere(self, radius=1., xpos=0., ypos=0., zpos=0., zscale=0.8):
        # generate random position in a sphere
        phi = self.np_random.uniform(0, 2 * math.pi)
        costheta = self.np_random.uniform(-1, 1)
        gamma = self.np_random.uniform(0, 1)

        theta = math.acos(costheta)
        r = radius * gamma ** (1. / 3)

        x = r * math.sin(theta) * math.cos(phi) + xpos
        y = r * math.sin(theta) * math.sin(phi) + ypos
        z = r * zscale * math.cos(theta) + zpos

        position = np.array((x, y, z))

        # generate random rotation quaternion
        u = self.np_random.uniform(0, 1, size=(3,))
        rotation = np.array((math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1]),
                             math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1]),
                             math.sqrt(u[0]) * math.sin(2 * math.pi * u[2]),
                             math.sqrt(u[0]) * math.cos(2 * math.pi * u[2])))

        return position, rotation

    # GoalEnv Methods
    def goal_distance_xyz(self, achieved_goal, desired_goal):
        """
        Calculates closeness to desired goal from achieved goal using l2norm for position and quaternion rotation distance for rotation
        :param achieved_goal: The goal we have currently achieved
        :param desired_goal: The goal we want to achieve
        :return: float32 of distance to goal
        """
        position_distance = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        rotation_distance = np.arccos(np.square(achieved_goal[3:].dot(desired_goal[3:])) - 1)  # Formula for rotation between two quaternions
        return position_distance, rotation_distance

    def goal_distance_joint(self, achieved_goal, desired_goal):
        """
        Calculates closeness to desired goal from achieved goal using distance between joint angles
        :param achieved_goal: The goal we have currently achieved
        :param desired_goal: The goal we want to achieve
        :return: float32 of distance to goal
        """
        return np.linalg.norm(achieved_goal - desired_goal)

    def ball_distance(self, achieved_goal):
        """
        Calculates closeness to desired goal from ball using l2norm for position and distance between cup->ball vector and cup frame y axis
        :param achieved_goal: The goal we have currently achieved
        :return: float32 of distance to goal
        """
        position_distance = np.linalg.norm(achieved_goal[0:3] - self.sim.data.get_body_xpos('ball'))

        cup_to_ball_w = self.sim.data.get_body_xpos('ball') - self.sim.data.get_body_xpos('cup')
        cup_to_ball_unit_w = cup_to_ball_w/np.linalg.norm(cup_to_ball_w)
        cup_quat = Quaternion(achieved_goal[3:])
        C_cw = cup_quat.rotation_matrix
        cup_y_w = np.linalg.inv(C_cw) @ np.array([0, 1, 0])
        rotation_distance = scipy.spatial.distance.cosine(cup_y_w, cup_to_ball_unit_w)
        # rotation_distance = np.arccos(np.square(achieved_goal[3:].dot(desired_orientation)) - 1)  # Formula for rotation between two quaternions
        return position_distance, rotation_distance

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Computes weighted reward for achieved goal and desired goal
        :param achieved_goal: The goal we have currently achieved
        :param desired_goal: The goal we want to achieve
        :param info: Optional object containing additional information
        :return: float32 of weighted reward
        """
        # position_distance, rotation_distance = self.goal_distance(achieved_goal, desired_goal)
        # return self.pos_rew_weight * math.exp(-10*position_distance) + self.rot_rew_weight * math.exp(-10*rotation_distance)

        # position_distance = np.linalg.norm(achieved_goal[0:3] - self.sim.data.get_body_xpos('ball'))

        # position_distance, rotation_distance = self.ball_distance(achieved_goal)
        # return self.pos_rew_weight * math.exp(-10*position_distance) + self.rot_rew_weight * math.exp(-10*rotation_distance)
        if self.temp_penalty_type == 0:
            rew = 0.1 * (-self.goal_distance_joint(achieved_goal[0:6], desired_goal))
        elif self.temp_penalty_type == 1:
            rew = 0.1 * (-self.goal_distance_joint(achieved_goal[0:6], desired_goal)) + 0.01 * (-np.linalg.norm(achieved_goal[6:12]))
        elif self.temp_penalty_type == 2:
            rew = 0.1 * (-self.goal_distance_joint(achieved_goal[0:6], desired_goal)) + 0.001 * (-np.linalg.norm(self.sim.data.qacc[5:11]))
        elif self.temp_penalty_type == 3:
            rew = 0.1 * (-self.goal_distance_joint(achieved_goal[0:6], desired_goal)) + 0.01 * (-np.linalg.norm(achieved_goal[6:12])) + 0.001 * (-np.linalg.norm(self.sim.data.qacc[5:11]))

        return rew


    # Catching calculation
    def get_optimal_catching_pos(self):
        self.nlopt_opt.reset()
        # ur10_pos = self._get_arm_joint(self.sim)[0:6]
        # base_pos = np.hstack((self._get_base_joint(self.sim)[0:2], np.array([0])))

        opt_var = nlopt_functions.OptimizationVariables()

        ball_state = opt_var.current_ball_state
        ball_state.position = self.sim.data.get_body_xpos('ball')
        ball_state.velocity = self.sim.data.get_body_xvelp('ball')
        ball_state.current_time = self.sim.get_state().time
        ball_state.xcoeff = [0.0, ball_state.velocity[0], ball_state.position[0]]
        ball_state.ycoeff = [0.0, ball_state.velocity[1], ball_state.position[1]]
        ball_state.zcoeff = [-4.905, ball_state.velocity[2], ball_state.position[2]]
        ball_state.coeff_start_time = 0

        for i in range(6):
            opt_var.current_ur10_state[i].pos = self.sim.data.get_joint_qpos(self.arm_joint_names[i])
        for i in range(2):
            opt_var.current_base_state[i].pos = self.sim.data.get_joint_qpos(self.base_joint_names[i])
        opt_var.current_base_state[2].pos = 0

        self.nlopt_success = self.nlopt_opt.start_optimization(opt_var)
        sol = self.nlopt_opt.get_solution()

        return sol

    # CURRENTLY RETURNS NEW BALL POS
    def get_optimal_ball_trajectory(self):
        '''
        Calculate the optimal ball trajectory such that the ball will land optimally in the
        the current cup location/orientation
        :return: np.array of ball position at each timestep of optimal trajectory
        '''
        cup_pos = self.sim.data.get_body_xpos('cup')
        cup_quat = Quaternion(self.sim.data.get_body_xquat('cup'))
        cup_rot = cup_quat.rotation_matrix
        cup_y_axis = cup_rot @ np.array([0, 1, 0])
        cup_y_axis *= np.clip(np.random.normal(1, .2, cup_y_axis.shape), 0, 2)  # Add noise
        gravity = np.array([0, 0, -9.81])

        ball_speed = np.abs(4 + 0.3*np.random.randn())
        travel_time = np.abs(0.7 + 0.1*np.random.rand())
        print(f"ball_speed: {ball_speed} | travel_time: {travel_time}")

        new_ball_pos = cup_pos + cup_y_axis * ball_speed * travel_time + 0.5 * gravity * travel_time ** 2
        new_ball_vel = cup_y_axis * ball_speed + gravity * travel_time

        return new_ball_pos, -new_ball_vel

    # test if point testpt is within a cylinder defined by two points, length of cylinder, and radius
    # Mohamed: I think this function only really checked if the ball was in the bottom half
    # plus it was hard ot understand
    def is_point_in_cylinder(self, pt_1: np.array, pt_2: np.array, length_squared: float, radius_squared: float, pt_test: np.array):
        d_pt = pt_2 - pt_1
        d_pt_test = pt_test - pt_1
        dot = np.dot(d_pt, d_pt_test)

        if dot < 0 or dot > length_squared:
            return False
        else:
            cyl_axes_dist_sq = np.dot(d_pt_test, d_pt_test) - dot**2/length_squared
            if cyl_axes_dist_sq > radius_squared:
                return False
            else:
                return True
    def is_point_in_cylinder(self, center, bottom, length, radius, pt_test):
        d = (pt_test - bottom) # pointing to point
        n = (center - bottom) # pointing inside the cylinder - axis of cylinder

        n = n / np.sqrt(np.dot(n, n))

        d_along_axis = np.dot(d, n) # distance in direction of axis

        if d_along_axis < 0: # the point is in the opposite direction of the center of the cylinder
            return False

        if d_along_axis > length: # the point is outside on the other side of the cylinder
            return False

        # horizontal direction
        d_perp_to_axis_squared = np.dot(d, d) - d_along_axis**2 # strictly nonnegative

        if d_perp_to_axis_squared > radius ** 2:
            return False

        return True

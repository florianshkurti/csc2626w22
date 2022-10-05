"""
Author: Ke Dong(kedong0810@gmail.com)
Date: 2019-12-18
Brief: nlopt cost and constraints functions
"""

import numpy as np
import copy

from gym_thing.nlopt_optimization.utility import State1D, BallStateCoeff, PhysicalLimits
import gym_thing.nlopt_optimization.utility as U
import gym_thing.kinematics.kinematics as K

BASE_THETA = 0.0
NUM_UR10_JOINTS = 6
NUM_BASE_JOINTS = 3
FINITE_DIFFERENCE = 1e-4


class NloptData(object):
    """Data structure for auxiliary nlopt setting data
    """
    def __init__(self):
        self.current_ball_state = BallStateCoeff()
        self.current_ur10_state = [State1D() for _ in range(NUM_UR10_JOINTS)]
        self.current_base_state = [State1D() for _ in range(NUM_BASE_JOINTS)]
        self.cost_weight_params = [1.0, 5.0]
        self.base_limits = PhysicalLimits(NUM_BASE_JOINTS)
        self.ur10_limits = PhysicalLimits(NUM_UR10_JOINTS)
        self.finite_difference = FINITE_DIFFERENCE


class OptimizationVariables(object):
    """Data structure to pack optimization related variables
    """
    def __init__(self):
        self.current_ball_state = BallStateCoeff()
        self.current_ur10_state = [State1D() for _ in range(NUM_UR10_JOINTS)]
        self.current_base_state = [State1D() for _ in range(NUM_BASE_JOINTS)]
        self.goal_ur10_state = [State1D() for _ in range(NUM_UR10_JOINTS)]
        self.goal_base_state = [State1D() for _ in range(NUM_BASE_JOINTS)]
        self.duration = 0.0


def reach_check_ramp(q_start, q_end, qv_start, vel_max, acc_max, t):
    """Calculates whether q_end is reachable from current position q_start

    Args:
        q_start: scalar, current position
        q_end:  scalar, desired position
        qv_start: scalar, current velocity
        vel_max: scalar, max velocity
        acc_max: scalar, max acceleration
        t: scalar, trajectory duration
        constraints: list, to be populated with distance residuals

    Returns:
        dist_res
    """
    dist_to_travel = q_end - q_start
    if dist_to_travel > 0:
        dist_max = abs(vel_max) * t - (vel_max - qv_start) * (vel_max - qv_start) / (2 * abs(acc_max))
    else:
        dist_max = abs(vel_max) * t - (vel_max + qv_start) * (vel_max + qv_start) / (2 * abs(acc_max))
    return abs(dist_to_travel) - dist_max


def ur10_collision_check(config):

    tcp_frame = K.forward_kinematics_arch2tcp(config)
    tcp_position = [tcp_frame[0, 3], tcp_frame[1, 3], tcp_frame[2, 3]]

    return min(tcp_position[2] + 0.25, tcp_position[0] + 0.05)


class NloptFunctions(object):
    """A class that combines nlopt data and functions
    """
    def __init__(self):
        self.nlopt_data = NloptData()

    def set_ball_state(self, ball_state):
        self.nlopt_data.current_ball_state = copy.copy(ball_state)

    def set_ur10_state(self, ur10_state):
        self.nlopt_data.current_ur10_state = copy.copy(ur10_state)

    def set_base_state(self, base_state):
        self.nlopt_data.current_base_state = copy.copy(base_state)

    def nlopt_cost_helper(self, x):
        """Calculates the cost of a solution vector

        Args:
            x: a solution vector

        Returns:
            cost: the cost
        """
        move_ur10, move_base = 0.0, 0.0

        for i, state in enumerate(self.nlopt_data.current_ur10_state):
            move_ur10 += np.square(x[i] - state.pos)
        move_ur10 = move_ur10 / NUM_UR10_JOINTS

        for i in range(NUM_BASE_JOINTS-1):
            move_base += np.square(x[i + NUM_UR10_JOINTS] - self.nlopt_data.current_base_state[i].pos)
        move_base = move_base / (NUM_BASE_JOINTS - 1)

        cost = move_ur10 * self.nlopt_data.cost_weight_params[0] + move_base * self.nlopt_data.cost_weight_params[1]

        return cost

    def nlopt_cost(self, x, grad):
        """Calculates the cost of a solution vector and the cost function's gradient

        Args:
            x: a solution vector
            grad: graident vector to

        Returns:
            cost: the cost
        """
        cost = self.nlopt_cost_helper(x)
        x_copy = copy.copy(x)

        if grad.size > 0:
            for i in range(grad.size):
                x_copy[i] += self.nlopt_data.finite_difference
                cost_positive = self.nlopt_cost_helper(x_copy)
                x_copy[i] -= 2 * self.nlopt_data.finite_difference
                cost_negative = self.nlopt_cost_helper(x_copy)
                x_copy[i] += self.nlopt_data.finite_difference
                grad[i] = (cost_positive - cost_negative) / (2 * self.nlopt_data.finite_difference)

        return cost

    def nlopt_equality_helper(self, results, x):
        """Calculates the equality constraints

        Args:
            results: a vector to be populated with equlaity equation residuals
            x: a solution vector

        Returns:
            None
        """
        time = x[-1]
        x_copy = copy.copy(x)
        x_copy[-1] = BASE_THETA

        tcp_frame = K.forward_kinematics_odom2tcp(x_copy)
        tcp_position = np.array([tcp_frame[0, 3], tcp_frame[1, 3], tcp_frame[2, 3]])
        tcp_orientation = np.array([-tcp_frame[0, 1], -tcp_frame[1, 1], -tcp_frame[2, 1]])

        ball_pos, ball_vel = U.ball_dynamics_coeff(self.nlopt_data.current_ball_state, time)
        results[0] = U.l2_norm(tcp_position, ball_pos)
        results[1] = U.orientation_norm(tcp_orientation, ball_vel)

    def nlopt_equality(self, results, x, grad):
        """Calculates the equality constraints for position and orientation matching

        Args:
            results: a vector to be populated with equlity equation residuals
            x: a solution vector
            grad: a vector to be populated with gradients
        Returns:
            None
        """
        x_copy = copy.copy(x)
        self.nlopt_equality_helper(results, x)

        n, m = len(x), len(results)
        if grad.size > 0:
            results_positive, results_negative = copy.copy(results), copy.copy(results)
            for i in range(n):
                x_copy[i] += self.nlopt_data.finite_difference
                self.nlopt_equality_helper(results_positive, x_copy)
                x_copy[i] -= 2 * self.nlopt_data.finite_difference
                self.nlopt_equality_helper(results_negative, x_copy)
                x_copy[i] += self.nlopt_data.finite_difference
                for j in range(m):
                    grad[j, i] = \
                        (results_positive[j] - results_negative[j]) \
                        / (2 * self.nlopt_data.finite_difference)

    def nlopt_inequality_helper(self, results, x):
        """ Calculates the inequality constraints for reachability check

        Args:
            results: a vector to be populated with inequality equation residuals
            x: a solution vector

        Returns:
            None
        """
        t = x[-1]

        for i in range(NUM_UR10_JOINTS):
            results[i] = reach_check_ramp(self.nlopt_data.current_ur10_state[i].pos,
                                          x[i],
                                          self.nlopt_data.current_ur10_state[i].vel,
                                          self.nlopt_data.ur10_limits.vmax[i],
                                          self.nlopt_data.ur10_limits.amax[i],
                                          t)
        for i in range(NUM_BASE_JOINTS - 1):
            index = i + NUM_UR10_JOINTS
            results[index] = reach_check_ramp(self.nlopt_data.current_base_state[i].pos,
                                              x[index],
                                              self.nlopt_data.current_base_state[i].vel,
                                              self.nlopt_data.base_limits.vmax[i],
                                              self.nlopt_data.base_limits.amax[i],
                                              t)
        results[-1] = - ur10_collision_check(x)

    def nlopt_inequality(self, results, x, grad):

        x_copy = copy.copy(x)
        self.nlopt_inequality_helper(results, x)

        n, m = len(x), len(results)
        if grad.size > 0:
            results_positive, results_negative = copy.copy(results), copy.copy(results)
            for i in range(n):
                x_copy[i] += self.nlopt_data.finite_difference
                self.nlopt_inequality_helper(results_positive, x_copy)
                x_copy[i] -= 2 * self.nlopt_data.finite_difference
                self.nlopt_inequality_helper(results_negative, x_copy)
                x_copy[i] += self.nlopt_data.finite_difference
                for j in range(m):
                    grad[j, i] = (results_positive[j] - results_negative[j]) \
                                      / (2 * self.nlopt_data.finite_difference)






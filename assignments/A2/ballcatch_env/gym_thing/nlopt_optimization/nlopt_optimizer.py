"""
Author: Ke Dong(kedong0810@gmail.com)
Date: 2019-12-18
Brief: nlopt optimizer related functions
"""

import json
import numpy as np
import copy
import sys

import nlopt

import gym_thing.nlopt_optimization.nlopt_functions as F
import gym_thing.nlopt_optimization.utility as U

def get_physical_limits(config, limits):

    limits.pmin = np.array(config['pmin'])
    limits.pmax = np.array(config['pmax'])
    limits.vmin = np.array(config['vmin'])
    limits.vmax = np.array(config['vmax'])
    limits.amin = np.array(config['amin'])
    limits.amax = np.array(config['amax'])


class NloptOptimizer(object):

    def __init__(self, config_pth = None):
        """Constructs an instance according to config_pth

        Args:
            config_pth: a json configuration file
        """

        # pre-allocate parameters
        self._ur10_joints = 6
        self._base_joints = 3
        self._w_ur10_move = 1.0
        self._w_base_move = 5.0

        self._ur10_limits = U.PhysicalLimits(self._ur10_joints)
        self._base_limits = U.PhysicalLimits(self._base_joints)

        self._nlopt_stop_equality_res = 5e-2  # how close to equality condition to consider it satisfied
        self._nlopt_stop_inequality_res = 1e-3
        self._nlopt_stop_max_time = 1
        self._time_lower_bound = 0.4
        self._finite_difference = 1e-4

        # parse configuration parameters for the above parameters
        if config_pth is not None:
            self.parse_configuration(config_pth)

        self._n_joints = self._ur10_joints + self._base_joints

        # set up nlopt related parameters
        self._nlopt_var_min_range = np.array(self._ur10_limits.pmin.tolist() + self._base_limits.pmin.tolist())
        self._nlopt_var_min_range[-1] = self._time_lower_bound
        self._nlopt_var_max_range = np.array(self._ur10_limits.pmax.tolist() + self._base_limits.pmax.tolist())
        self._nlopt_var_max_range[-1] = 5

        self._cost_weight = np.array([self._w_ur10_move, self._w_base_move])

        self._nlopt_tol_equality = np.array([self._nlopt_stop_equality_res] * 2)
        self._nlopt_tol_inequality = np.array([self._nlopt_stop_inequality_res] * self._n_joints)

        self.nlopt_solution = np.array([0.0] * self._n_joints)
        self._nlopt_solution_previous = np.array([0.0] * self._n_joints)

        self._nlopt_functions = F.NloptFunctions()
        self._nlopt_functions.nlopt_data.cost_weight_params = copy.copy(self._cost_weight)
        self._nlopt_functions.nlopt_data.base_limits = copy.copy(self._base_limits)
        self._nlopt_functions.nlopt_data.ur10_limits = copy.copy(self._ur10_limits)
        self._nlopt_functions.nlopt_data.finite_difference = copy.copy(self._finite_difference)

        self._flag_init_sol = False

    def parse_configuration(self, config_pth):
        """Parses configuration file

        Args:
            config_pth: a json file path

        Returns:
            None
        """
        with open(config_pth, 'r') as fid:
            config = json.load(fid)

        self._ur10_joints = int(config['ur10_joints'])
        self._base_joints = int(config['base_joints'])
        self._w_ur10_move = config['weight']['ur10_move']
        self._w_base_move = config['weight']['base_move']

        get_physical_limits(config['ur10_limits'], self._ur10_limits)
        get_physical_limits(config['base_limits'], self._base_limits)

        self._nlopt_stop_equality_res = config['nlopt']['stop_equality_res']
        self._nlopt_stop_inequality_res = config['nlopt']['stop_inequality_res']
        self._nlopt_stop_max_time = config['nlopt']['stop_max_time']
        self._time_lower_bound = config['nlopt']['time_lower_bound']
        self._finite_difference = config['nlopt']['finite_difference']

    def get_solution(self):
        return copy.copy(self._nlopt_solution_previous)

    def start_optimization(self, opt_var:F.OptimizationVariables):
        """Constructs a nlopt optimizer and starts optimization

        Args:
            opt_var: an OptimizationVariables instance

        Returns:
            flag: True if optimization successes
        """
        self._nlopt_functions.set_ball_state(opt_var.current_ball_state)
        self._nlopt_functions.set_ur10_state(opt_var.current_ur10_state)
        self._nlopt_functions.set_base_state(opt_var.current_base_state)

        if not self._flag_init_sol:
            # use current states for initial guess
            self._nlopt_solution_previous = [state.pos for state in opt_var.current_ur10_state]\
                                            + [state.pos for state in opt_var.current_base_state[:-1]]\
                                            + [0.5]

        # max duration to search for catch
        self._nlopt_var_max_range[-1] = 5#U.get_max_duration(opt_var.current_ball_state.position[2], opt_var.current_ball_state.velocity[2])

        opt = nlopt.opt(nlopt.LD_SLSQP, self._n_joints)

        opt.set_lower_bounds(self._nlopt_var_min_range)
        opt.set_upper_bounds(self._nlopt_var_max_range)

        opt.set_min_objective(self._nlopt_functions.nlopt_cost)

        # opt.add_inequality_mconstraint(self._nlopt_functions.nlopt_inequality, self._nlopt_tol_inequality)
        opt.add_equality_mconstraint(self._nlopt_functions.nlopt_equality, self._nlopt_tol_equality)

        opt.set_ftol_rel(self._nlopt_stop_inequality_res)
        opt.set_maxtime(self._nlopt_stop_max_time)

        self.nlopt_solution = copy.copy(self._nlopt_solution_previous)

        task_constraints = np.array([5.0, 5.0])

        try:
            xopt = opt.optimize(self.nlopt_solution)
            # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            # print(f"NLOPT SOLUTION: {xopt}")
            self._nlopt_functions.nlopt_equality_helper(task_constraints, xopt)
            if task_constraints[0] > self._nlopt_stop_equality_res * 5:
                # print("EQUALITY CONSTRAINTS FAILED")
                return False

            self._nlopt_solution_previous = copy.copy(xopt)
            self._flag_init_sol = True

        except:
            print("error: ", sys.exc_info()[0])
            return False

        return True

    def reset(self):
        self._flag_init_sol = False
        self.nlopt_solution = np.array([0.0] * self._n_joints)
        self._nlopt_solution_previous = np.array([0.0] * self._n_joints)
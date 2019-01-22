import numpy as np
from math import pi

class PID(object):
    def __init__(self, kp, ki, deriv_prediction_dt, max_deriv_noise_gain, max_window_size=3):
        self.kp = kp
        self.kd = kp * deriv_prediction_dt
        self.ki = ki

        assert (max_window_size >= 3)
        
        self.timestamps_of_errors = []   # in secs
        self.bounded_window_of_errors = []
        self.bounded_window_of_error_derivs = []
        
        self.max_window_size = max_window_size
        self.max_deriv_noise_gain = max_deriv_noise_gain
        self.deriv_prediction_dt = deriv_prediction_dt
        self.integral_of_errors = 0
        self.control = 0

        self.alpha = None    # weight [0,1] on the previous estimate of derivative. 0 ignore previous estimate
        
        
    def erase_history(self):
        self.timestamps_of_errors = []  
        self.bounded_window_of_errors = []
        self.bounded_window_of_error_derivs = []
        self.integral_of_errors = 0
        self.control = 0
        
        
    def set_params(self, kp, ki, deriv_prediction_dt, max_deriv_noise_gain, alpha=None):
        self.kp = kp
        self.ki = ki
        self.kd = kp * deriv_prediction_dt
        self.max_deriv_noise_gain = max_deriv_noise_gain
        self.deriv_prediction_dt = deriv_prediction_dt
        self.alpha = alpha
        
    def is_initialized(self):
        W = len(self.bounded_window_of_errors)
        return (W >= self.max_window_size)

    
    def compute_error_derivative(self):
        assert (len(self.bounded_window_of_errors) >= 2)
        dt1 = self.timestamps_of_errors[-1] - self.timestamps_of_errors[-2]
        assert (dt1 > 0)
        
        curr_error_diff = self.bounded_window_of_errors[-1] - self.bounded_window_of_errors[-2]

        if self.bounded_window_of_error_derivs:
            prev_deriv = self.bounded_window_of_error_derivs[-1]
            alpha, beta = 0, 1

            if self.alpha is None and (self.deriv_prediction_dt > 0 or self.max_deriv_noise_gain > 0):
                # See Astrom's book "Automatically Tuning PID Controllers, page 21"
                alpha = self.deriv_prediction_dt / (self.deriv_prediction_dt + self.max_deriv_noise_gain * dt1)
                beta = self.kp * self.max_deriv_noise_gain * alpha
                
            elif self.alpha:
                # Relative weights are manually set 
                alpha = self.alpha
                beta = (1.0 - self.alpha)/dt1 

            error_derivative = alpha * prev_deriv + beta * curr_error_diff
            return error_derivative

        else:
            return curr_error_diff / dt1
        
        

    def compute_error_integral(self):
        dt1 = self.timestamps_of_errors[-1] - self.timestamps_of_errors[-2]
        assert (dt1 > 0)
        curr_error = self.bounded_window_of_errors[-1]
        return curr_error * dt1

    
    def update(self, error, timestamp):
        self.bounded_window_of_errors.append(error)
        self.timestamps_of_errors.append(timestamp)
        
        W = len(self.bounded_window_of_errors)
        if (W > self.max_window_size):
            self.bounded_window_of_errors = self.bounded_window_of_errors[1:]
            self.timestamps_of_errors = self.timestamps_of_errors[1:]

                
        if W > 1:
            error_derivative = self.compute_error_derivative()
            self.bounded_window_of_error_derivs.append(error_derivative)
            if (len(self.bounded_window_of_error_derivs) > self.max_window_size - 1):
                self.bounded_window_of_error_derivs = self.bounded_window_of_error_derivs[1:]

                    
        if self.is_initialized():
            curr_error = self.bounded_window_of_errors[-1]
            error_derivative = self.bounded_window_of_error_derivs[-1]
            curr_error_integral = self.compute_error_integral()
        
            self.integral_of_errors += curr_error_integral
            self.control = self.kp*curr_error + self.kd*error_derivative + self.ki*self.integral_of_errors 

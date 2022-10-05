import numpy as np

GRAVITY = 9.81

class State1D(object):
    def __init__(self):
        self.pos = 0.0
        self.vel = 0.0
        self.acc = 0.0


class BallStateCoeff(object):
    def __init__(self):
        self.current_time = 0.0
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.xcoeff = np.array([0.0, 0.0, 0.0])
        self.ycoeff = np.array([0.0, 0.0, 0.0])
        self.zcoeff = np.array([0.0, 0.0, 0.0])
        self.coeff_start_time = 0.0


class PhysicalLimits(object):
    def __init__(self, dim:int):
        self.pmin = np.array([0.0] * dim)
        self.pmax = np.array([0.0] * dim)
        self.vmin = np.array([0.0] * dim)
        self.vmax = np.array([0.0] * dim)
        self.amin = np.array([0.0] * dim)
        self.amax = np.array([0.0] * dim)


def ball_dynamics_coeff(current_ball_state:BallStateCoeff, t):
    """ Predicts ball's future position and velocity in future

    Args:
        current_ball_state: current ball state
        t: a duration time

    Returns:
        ball_pos: future ball positions
        ball_vel: future ball velocity
    """
    time = current_ball_state.current_time + t - current_ball_state.coeff_start_time

    ball_pos = np.array([0.0, 0.0, 0.0])
    ball_vel = np.array([0.0, 0.0, 0.0])

    ball_pos[0] = current_ball_state.xcoeff[0] * time * time + current_ball_state.xcoeff[1] * time\
                  + current_ball_state.xcoeff[2]
    ball_vel[0] = 2 * current_ball_state.xcoeff[0] * time + current_ball_state.xcoeff[1]

    ball_pos[1] = current_ball_state.ycoeff[0] * time * time + current_ball_state.ycoeff[1] * time\
                  + current_ball_state.ycoeff[2]
    ball_vel[1] = 2 * current_ball_state.ycoeff[0] * time + current_ball_state.ycoeff[1]

    ball_pos[2] = current_ball_state.zcoeff[0] * time * time + current_ball_state.zcoeff[1] * time\
                  + current_ball_state.zcoeff[2]
    ball_vel[2] = 2 * current_ball_state.zcoeff[0] * time + current_ball_state.zcoeff[1]

    return ball_pos, ball_vel


def l2_norm(x1, x2):
    # return np.sqrt(np.sum(np.square(np.array(x1) - np.array(x2))))
    return np.linalg.norm(x1 - x2)

def orientation_norm(v1, v2):
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)

    dot_product_sum = np.sum(np.multiply(v1_arr, v2_arr))
    return 1 - dot_product_sum / (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr))


def get_max_duration(pos_z, vel_z):
    if abs(vel_z) < 1e-3 and abs(GRAVITY) < 1e-3:
        # test case, static ball and no gravity
        return 1e10
    elif abs(GRAVITY) < 1e-3:
        return abs(pos_z / vel_z)
    else:
        return (vel_z + np.sqrt(vel_z * vel_z + 2 * pos_z * GRAVITY)) / GRAVITY


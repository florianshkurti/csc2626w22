class BallCatchDMPPolicy:
    def __init__(self, dmp, dt):
        """
        dmp: A trained DMP for planning catch motions in joint space
        dt: the duration of a simulation timestep
        """
        self.dmp = dmp
        self.g = None
        self.history = []
        self.plan = []
        self.dt = dt
        self.tau = 0.6
    
    def set_goal(self, state, goal):
        self.time_step = -1
        self.x0 = state[:6]
        self.g = goal
        self.history = []
    
    def select_action(self, state):
        """
        Returns the next action (desired joint angle displacement for the UR10)

        state: The current state. An array of dimension (21,)
            Elements
                0-6: joint angles at time t
                6-12: joint angle velocities at time t
                12-15: ball position in world coordinates at time t
                15-18: ball position in world coordinates at time t-1
                18-21: ball position in world coordinates at time t-2
        """
        self.time_step += 1
        time = self.time_step * self.dt
        x = state[:6]
        x_dot = state[6:12]
        
        # TODO: Query the DMP
        x_target, x_dot_target = 

        self.history.append(state)
        self.plan.append(x_target)
        return x_dot_target - x_dot


import gym
from gym import error, spaces, utils
from car_racing import *
from pid import PID
import numpy as np
from math import atan2, cos, sin, sqrt

class FullStateCarRacingEnv(CarRacing):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=0):
        CarRacing.__init__(self, seed=seed)

        self.orientation_pid = PID(kp=1.5/3.14, ki=0, deriv_prediction_dt=10.0/FPS, max_deriv_noise_gain=3)
        self.distance_pid = PID(kp=0.25/2, ki=0, deriv_prediction_dt=100.0/FPS, max_deriv_noise_gain=60, max_window_size=20)
        self.gas_pid = PID(kp=0.3, ki=0, deriv_prediction_dt=10.0/FPS, max_deriv_noise_gain=3)

        self.target_speed = 30.0
        self.max_gas = 0.05
        self.draw_target_nav_frame = False
        
    def point_segment_dist(self, p, a, b):
        n = b - a
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            return np.linalg.norm(p - a)

        n = n / norm_n
        ap = a - p
        proj_on_line = ap.dot(n) * n
        
        if np.linalg.norm(proj_on_line) > norm_n:
            return min(np.linalg.norm(p - a), np.linalg.norm(p - b))
        
        return np.linalg.norm(ap - proj_on_line)
        
    def get_cross_track_error(self, car, track):
        # steer in [-1, 1], gas in [0, 1], break in [0 ,1]
        pld_min = np.finfo(np.float).max
        dest_min = 0

        p = car.hull.position
        p = np.array([p[0], p[1]])
        
        for i in range(1, len(track)):
            ai = np.array([track[i-1][2], track[i-1][3]])
            bi = np.array([track[i][2], track[i][3]])
            pld = self.point_segment_dist(p, ai, bi)
            if pld < pld_min:
                pld_min = pld 
                dest_min = i
        
        target_heading = track[dest_min][1] 
        error_heading = target_heading - car.hull.angle
        error_heading =  atan2(sin(error_heading), cos(error_heading)) 
        
        R_world_trackframe = np.array([ [cos(target_heading), sin(target_heading)],
                                        [-sin(target_heading), cos(target_heading)] ])

        p_trackframe_world = np.array( track[dest_min][2:4] ).reshape((2,1))
        p_car_world = np.array( [car.hull.position[0], car.hull.position[1]] ).reshape((2,1))

        p_car_trackframe = R_world_trackframe.dot(p_car_world - p_trackframe_world) 
        error_dist = p_car_trackframe[0][0]

        #print (error_heading * 180.0 / 3.14, error_dist, p_car_trackframe[1][0])
        return error_heading, error_dist, dest_min 

    
    def get_expert_action(self, car, track):
        current_speed = sqrt(self.car.hull.linearVelocity.x**2 + self.car.hull.linearVelocity.y**2)
        eh, ed, _ = self.get_cross_track_error(car, track)
        
        es = self.target_speed - current_speed
        
        self.orientation_pid.update(eh, self.t)
        self.distance_pid.update(ed, self.t)
        self.gas_pid.update(es, self.t)

        a1 = np.array( [-self.distance_pid.control, 0, 0] )
        a2 = np.array( [-self.orientation_pid.control, 0, 0] )
        a = 0.2*a1 + 0.8*a2
        #a = a1

        a[0] = max(a[0], -1.0)
        a[0] = min(a[0],  1.0)
        
        if (self.gas_pid.control > 0):
            a[1] = min(self.gas_pid.control, self.max_gas)  # positive gas
        else:
            a[2] = min(-self.gas_pid.control, 1.0)  # positive brake
            
        return a, eh, ed

    def step(self, action):
        
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS
        
        self.state = self.render("state_pixels")
        
        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1

            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.tile_visited_count==len(self.track):
                done = True

            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        expert_action, eh, ed = self.get_expert_action(self.car, self.track)
        if (abs(ed) > 10):
            print ('Learner deviated too far!')
            done = True
            
        return self.state, expert_action, step_reward, done, {}
    
    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")

        angle, x, y = self.track[0][1:4]
        self.car = Car(self.world, angle, x, y)
        
        
        return self.step(None)[0]

    def render(self, mode='human', do_render_indicators=False):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        #zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom = ZOOM*SCALE
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
            
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")
        
        error_heading, error_dist, dest_min = self.get_cross_track_error(self.car, self.track)

        if self.draw_target_nav_frame:
            track_size = len(self.track)
            a = self.track[dest_min][2:4]
            b = self.track[(dest_min + 1) % track_size][2:4]
            c = (b[0] - a[0], b[1] - a[1]) 
            self.viewer.draw_line(start=a, end=b)
            self.viewer.draw_line(start=a, end=(a[0] + c[1], a[1] - c[0]))
            
        
        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
            
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            if do_render_indicators:
                self.render_indicators(WINDOW_W, WINDOW_H)  

            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()

            if do_render_indicators:
                self.render_indicators(WINDOW_W, WINDOW_H)
                
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

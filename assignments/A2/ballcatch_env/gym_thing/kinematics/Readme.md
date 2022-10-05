## Usage
This directory provides forward kinematics functions of gym_thing. Note that we do NOT provide 
inverse kinematics functions (for now). Also note that the forward kinematics here is calibrated to match
the need of the ball catching project. For your use, you may want to change ``OFFSET_CALIBRATION`` to zeros
and ``OFFSET_K_TCP_Z`` to the value you want. Remember to change the correpsonding attribute of body
``calibration_matrix_1`` and ``thing_tool`` in file ``robot_with_gripper.xml`` file or other xml file that you use

### Major API functions:
- ``forward_kinematics_odom2tcp(theta)`` (recommended)
- ``forward_kinematics_arch2tcp(theta)``


## Relationship between c++ kinematics and python kinematics

(This is only for developers)

To better match with real-world robot kinematics functions, we add extra calibration matrices in our ROS kinematics functions.
The extra calibration parameter consists of three parts:    
    - TCP translational parameter: $\Delta x_{tcp}, \Delta y_{tcp}, \Delta z_{tcp}$  
    - Calibration transformation matrix 1: $x_1, y_1, z_1, rot_x_1, rot_y_1, rot_z_1$    
    - Calibration transformation matrix 2: $x_2, y_2, z_2, rot_x_2, rot_y_2, rot_z_2$   
   
Calbration Matrix 1 (referred as Cal-1) is inserted before arm joint transformations, while Calibration Matrix 2 (referred as Cal-2)
is inserted between arm joint transformations. However, since the mujoco xml file adds an extra 180 degree rotation of the arm base
joint, the DH transformation matrix we use for kinematics functions cannot match with transformations in the xml file precisely.

In other words, we can easily add Cal-1 to the mujoco xml file, but it is hard to add Cal-2. However, in our current ros implementation,
both Cal-1 and Cal-2 are used.

Calibration experiments show that calibration error using both Cal-1 and Cal-2 is slightly smaller than only using Cal-1 
(the difference is around 0.001). So we only use Cal-1 in this mujoco xml file.

As a result, we have the following diagram:
```
    kinematics.cpp(TCP, Cal-1, Cal-2)    <---error: around 0.020mm----->  kinematics.py (TCP, Cal-1)
            |                                                            /         |
            |                                                       /              |
        (error: around 0.010mm)       (error: around 0.011 mm)           (perfectly matching)
            |                          /                                           |        
            |                  /                                                   |
       Real robots                                                          Mujoco simulation
                  
```
            
     
    

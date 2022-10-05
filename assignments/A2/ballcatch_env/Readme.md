# Description
This repo provides infrastructures for thing mujoco simulation. It contains the following three sub-directory    

- ``gym_thing/envs``: mujoco simulation environment implementation    
    - We put two gym environments here: ``thing-v0`` is used for general simulation purpose, while ``ballcatch-v0`` is 
    specially designed for the ball catching task
- ``gym_thing/kinematics`` forward kinematics functions for the thing robot, specially calibrated for the ball catching task
- ``gym_thing/nlopt_optimization`` an nonlinear optimization solver and related functions only for the ball catching

# Requirement:

- ``gym``
- ``mujoco_py`` (also a valid mujoco license)
- ``pyquaternion``
- ``nlopt`` if you need to use nlopt (only needed if you want to use the ``ballcatch-v0`` environment)

# Usage:
You need to install the ``gym_thing`` package to register our customized environments with ``gym``. 

In the root directory:
```
    pip install -e .
```

One typical example is:
```python
import gym
import gym_thing

env = gym.make('thing-v0')
env.reset()
env.render()
```

Note that ``ballcatch-v0`` is specially designed for the ball catching task


# CSC2626 Assignment 2 - Dynamic Movement Primitives

##  Basic Setup
This project requires python 3.7. To install the python requirements:

    pip install -r requirments.txt
    pip install -e ballcatch_env/

### Install Mujoco:
The last thing you need is the mujoco simulator. You can install that on linux as follows:

Download mujoco 210 from https://github.com/deepmind/mujoco/releases/tag/2.1.0
Install by placing the unzipped folder "mujoco210" under ~/.mujoco/
    mkdir ~/.mujoco
    tar -xvf mujoco210-linux-x86_64.tar
    mv mujoco210 ~/.mujoco/


## Notes

If you try to run the GUI, you may get errors telling you to set LD_LIBRARY_PATH or LD_PRELOAD. If so, this should fix it:
    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin:/usr/lib/nvidia
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

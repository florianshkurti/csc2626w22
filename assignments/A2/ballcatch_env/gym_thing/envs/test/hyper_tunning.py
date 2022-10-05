import os
import sys
import uuid
import shutil
import numpy as np

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


import ray
from ray import tune
from ray.tune import Analysis
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import thing_dynamics_test as test

# Constant path and strings
ARM_ACTUATOR_NAME = ["ur10_arm_0_shoulder_pan_act", "ur10_arm_1_shoulder_lift_act", "ur10_arm_2_elbow_act",
                              "ur10_arm_3_wrist_1_act", "ur10_arm_4_wrist_2_act", "ur10_arm_5_wrist_3_act"]
BASE_ACTUATOR_NAME = ["ridgeback_x_act", "ridgeback_y_act"]
ACTUATOR_NAME = ARM_ACTUATOR_NAME + BASE_ACTUATOR_NAME


ROSBAG_PTH = "/h/kedong/Projects/ballcatch_env/gym_thing/envs/test/rosbag_data.npy"


def xml_file_modify(original_xml_file, output_xml_file, actuator_name, kp_value):
    """
    Modifies kp gain value in a robot description file

    :param original_xml_file: file path to the original xml file
    :param output_xml_file: file path to the output temporary xml file
    :param actuator_name: actuator name
    :param kp_value: kp value to be written
    :return: True if modification successes
    """

    tree = ET.parse(original_xml_file)
    root = tree.getroot()
    actuator = root.find("actuator")

    for child in actuator:
        name = child.get('name')
        if name == actuator_name:
            child.set('kp', str(kp_value))

    tree.write(output_xml_file)

    return True


def trainable(config):
    """
    Training function for the Tune library. Returns error for one specific configuration

    :param config:
    :return: error
    """
    print(config)
    kp_values = [config['arm_0'], config['arm_1'], config['arm_2'], config['arm_3'], config['arm_4'], config['arm_5'],
                 config['base_0'], config['base_1']]


    original_xml_file = "/h/kedong/Projects/ballcatch_env/gym_thing/envs/assets/robot_with_gripper.xml"
    test_xml_file = original_xml_file.split('.xml')[0] + '_' + str(uuid.uuid4()) + '.xml'
    shutil.copyfile(original_xml_file, test_xml_file)

    # rewrite xml file
    for index, kp_value in enumerate(kp_values):
        xml_file_modify(test_xml_file, test_xml_file, ACTUATOR_NAME[index], kp_value)

    # instantiate gym environments
    errors = test.dynamic_test_dir(ROSBAG_PTH, os.path.basename(test_xml_file))
    os.remove(test_xml_file)

    mean_loss = np.mean(errors)
    std_loss = np.std(errors)
    tune.track.log(mean_loss=mean_loss,
                   std_loss=std_loss)


def trainable_bayes(arm_0, arm_1, arm_2, arm_3, arm_4, arm_5, base_0, base_1):
    """
    Training function for the Tune library. Returns error for one specific configuration

    :param config:
    :return: error
    """
    kp_values = [arm_0, arm_1, arm_2, arm_3, arm_4, arm_5, base_0, base_1]


    original_xml_file = "/h/kedong/Projects/ballcatch_env/gym_thing/envs/assets/robot_with_gripper.xml"
    test_xml_file = original_xml_file.split('.xml')[0] + '_' + str(uuid.uuid4()) + '.xml'
    shutil.copyfile(original_xml_file, test_xml_file)

    # rewrite xml file
    for index, kp_value in enumerate(kp_values):
        xml_file_modify(test_xml_file, test_xml_file, ACTUATOR_NAME[index], kp_value)

    # instantiate gym environments
    errors = test.dynamic_test_dir(ROSBAG_PTH, os.path.basename(test_xml_file))
    os.remove(test_xml_file)

    mean_loss = np.mean(errors)
    std_loss = np.std(errors)
    return -mean_loss


def base_config_test():
    optimal_kp_value = []
    for i in range(2): # only the first five arm joints are used
        analysis = tune.run(
            trainable,
            name="base_kp_tune_" + str(i),
            config={
                "index": i+6,
                "kp": tune.grid_search(list(range(15000, 25000, 1000)))
            },
            resources_per_trial={
                "cpu": 1,
                "gpu": 1
            }
        )
        optimal_kp_value.append(analysis.get_best_config(metric="mean_loss", mode="min"))
        print("Best kp value for the %d-th joint is %.5f" %(i, optimal_kp_value[-1]["kp"]))

    print(optimal_kp_value)


def search_main():
    ray.init()
    search_space = {
        'arm_0': (4000, 6000),
        'arm_1': (1800, 4000),
        'arm_2': (1800, 4000),
        'arm_3': (1500, 3000),
        'arm_4': (4000, 6000),
        'arm_5': (500, 1500),
        'base_0': (20000, 30000),
        'base_1': (20000, 30000)
    }

    algo = BayesOptSearch(
        search_space,
        metric="mean_loss",
        mode="min",
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        })


    scheduler = AsyncHyperBandScheduler(metric="mean_loss", mode="min")

    tune.run(
        trainable,
        name='gym_thing_tuning',
        search_alg=algo,
        scheduler=scheduler,
        num_samples=1000,
        resources_per_trial = {
            "cpu": 8,
            "gpu": 1
        },
        verbose=2,
        config={}
    )


def bayesian_search_main():

    search_space = {
        'arm_0': (4000, 6000),
        'arm_1': (1800, 4000),
        'arm_2': (1800, 4000),
        'arm_3': (1500, 3000),
        'arm_4': (4000, 6000),
        'arm_5': (500, 1500),
        'base_0': (20000, 30000),
        'base_1': (20000, 30000)
    }

    optimizer = BayesianOptimization(
        f=trainable_bayes,
        pbounds=search_space,
        verbose=2
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        n_iter=2000,
        init_points=100
    )

    print(optimizer.max)


def post_processing(datadir, titlename, output_file):

    analysis = Analysis(datadir)
    df = analysis.dataframe()
    df = df[["config/kp", "mean_loss", "std_loss"]]
    df.set_index("config/kp", inplace=True)
    df.sort_index(inplace=True)
    data = df.to_numpy()
    kp_index = df.index.values

    plt.figure()
    plt.errorbar(kp_index, data[:, 0], yerr=data[:, 1])
    plt.grid()
    plt.xlabel("kp value")
    plt.ylabel("RMSE")
    plt.title(titlename)
    plt.savefig(output_file + ".png")

    with open(output_file + ".txt", 'w') as fid:
        fid.write("kp \t loss mean \t loss std\n")
        for i in range(len(kp_index)):
            line = str(kp_index[i]) + '\t' + str(data[i, 0]) + '\t' + str(data[i, 1]) + '\n'
            fid.write(line)


if __name__ == "__main__":
    bayesian_search_main()



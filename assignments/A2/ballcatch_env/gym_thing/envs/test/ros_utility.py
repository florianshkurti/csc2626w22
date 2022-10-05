import rosbag
import sys

import numpy as np
import os

from scipy.interpolate import interp1d


def read_rosbag(bagname, topic, duration=None, start_time=None):
    """Reads messages of specified topics inside a rosbag

    Args:
        bagname: a rosbag's file path
        topic: a topic to be read
        duration: a duration time during which messages are collected

    Returns:
        a list containing all messages in the specified topic. If duration
        is given, only message between [start_time, start_time + duration]
        are collected
    """
    bag = rosbag.Bag(bagname)
    msgs = []
    for topic, msg, _ in bag.read_messages(topics=topic):
        msgs.append(msg)
    if duration:
        msgs_filtered = []
        start_time = msgs[0].header.stamp.to_sec() if not start_time else start_time
        for msg in msgs:
            if msg.header.stamp.to_sec() < start_time + duration:
                msgs_filtered.append(msg)
        return msgs_filtered
    return msgs


def rmse(X):
    return np.sqrt(np.mean(np.square(X), axis=0))


def controlStateReader(bagname, start_time_bag=None):
    bag = rosbag.Bag(bagname)
    desired_ur10, actual_ur10, error_ur10, time_ur10 = [], [], [], []
    desired_rb, actual_rb, error_rb, time_rb = [], [], [], []

    for _, msg, _ in bag.read_messages(
            topics=['/vel_based_pos_traj_controller/state', '/vel_based_pos_traj_controller_inverse_dynamics/state']):
        desired_ur10.append(msg.desired)
        actual_ur10.append(msg.actual)
        error_ur10.append(msg.error)
        time_ur10.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)

    for topic, msg, t in bag.read_messages(topics=['/cart/ridgeback_cartesian_controller_inverse_dynamics/state',
                                                   '/cart/ridgeback_cartesian_controller/state']):
        desired_rb.append(msg.desired)
        actual_rb.append(msg.actual)
        error_rb.append(msg.error)
        time_rb.append(msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs)

    if not start_time_bag:
        start_time = float("inf")
        if time_ur10:
            start_time = min(time_ur10)
        if time_rb:
            start_time = min(start_time, min(time_rb))
    else:
        start_time = start_time_bag

    for i in range(len(time_ur10)):
        time_ur10[i] -= start_time
    for i in range(len(time_rb)):
        time_rb[i] -= start_time
    return desired_ur10, actual_ur10, error_ur10, time_ur10, desired_rb, actual_rb, error_rb, time_rb


def find_nearest(array, value, threshold=0.1):
    """
    Finds the value in array closest to value and return its index.
    Returns -1 if no value close enough to the value is found
    """
    array_new = np.abs(np.array(array) - value)
    err, index = np.min(array_new), np.argmin(array_new)
    if err > threshold:
        return -1
    else:
        return index


def get_file_names(dir_name, suffix, prefix=None):
    """ Returns all file names with suffix in the dir_name
    """
    list_name = []
    for file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, file)
        if os.path.isdir(file_path):
            continue
        elif os.path.splitext(file_path)[1] == suffix:
            if prefix:
                if file.startswith(prefix):
                    list_name.append(file_path)
            else:
                list_name.append(file_path)
    return list_name


def rename_files(dirname, suffix, prefix):
    list_name = get_file_names(dirname, suffix)
    for name in list_name:
        file_name = os.path.split(name)[1]
        new_path = os.path.join(dirname, prefix+"_" + file_name)
        os.rename(name, new_path)


def curve_error_rmse(curve_points_a, curve_points_b):
    """
    compare twos curves' rooted mean square error. Uses cubic curve to
    fit curve b.
    """
    x_a, y_a = curve_points_a
    x_b, y_b = curve_points_b
    bound = (min(x_b), max(x_b))
    f_b_interp = interp1d(x_b, y_b, kind='cubic')
    error = []
    for i in range(len(x_a)):
        x = x_a[i]
        if (x < bound[0] or x > bound[1]):
            continue
        y_a_i = y_a[i]
        y_b_i = f_b_interp(x)
        error.append(np.square(y_a_i - y_b_i))

    return np.sqrt(np.mean(error))


def sort_joint_traj_msgs(joint_traj_msgs):
    """
    Sorts desired joint trajectory messages according to time_from_start

    Args:
        joint_traj_msgs: a list of joint trajectory messages

    Returns:
        a np array in the following format:
        [
            [time, joint_0, joint_1, ..., joint_n]
            ...
            [time, joint_0, joint_1, ..., joint_n]
        ]
    """
    joint_traj = []

    for msg in joint_traj_msgs:

        # extract joint trajctory of current msg:
        joint_traj_current = []
        current_start_time = msg.header.stamp.to_sec()
        for point in msg.points:
            joint_traj_current.append(
                [point.time_from_start.to_sec() + current_start_time] + [pos for pos in point.positions])

        start_index = 0
        for i in range(len(joint_traj)):
            if joint_traj[i][0] > joint_traj_current[0][0]:
                # new trajectory starts here
                start_index = i
                break
        joint_traj = joint_traj[:start_index] + joint_traj_current

    return np.array(joint_traj)


def sort_thing_trajectories(bagname):
    """
    Sorts thing joint trajectories. Uses linear interpolation to get
    base joint states at ur10 joint header stamp time

    Args:
        bagname: a path to a rosbag
    Returns:
        thing_traj: a numpy array containting thing desired trajectory
    """
    # use the first ball_state msg for start time
    ball_msgs = read_rosbag(bagname, ["/ball_states"])
    traj_msgs = read_rosbag(bagname, ["/desired_trajectory_thing"])

    if len(ball_msgs) == 0:
        start_time = traj_msgs[0].header.stamp.to_sec() - 2
    else:
        start_time = ball_msgs[0].start_time

    ur10_traj_msg = [msg.traj_ur10 for msg in traj_msgs]
    ur10_traj = sort_joint_traj_msgs(ur10_traj_msg)

    base_traj_msg = [msg.traj_rb for msg in traj_msgs]
    base_traj = sort_joint_traj_msgs(base_traj_msg)

    ur10_traj[:, 0] = ur10_traj[:, 0] - start_time
    base_traj[:, 0] = base_traj[:, 0] - start_time

    base_fit_x = interp1d(base_traj[:, 0], base_traj[:, 1], kind='linear')
    base_fit_y = interp1d(base_traj[:, 0], base_traj[:, 2], kind='linear')
    base_fit_r = interp1d(base_traj[:, 0], base_traj[:, 3], kind='linear')
    thing_traj = []

    for i in range(len(ur10_traj)):
        t = ur10_traj[i, 0]
        point = ur10_traj[i, :].tolist() + [base_fit_x(t), base_fit_y(t), base_fit_r(t)]
        point[0] += start_time
        thing_traj.append(point)

    return np.array(thing_traj)


def sort_joint_states_msg(bagname):
    """
    Sorts actual thing joint states
    """
    ball_msgs = read_rosbag(bagname, ["/ball_states"])
    if len(ball_msgs) == 0:
        traj_msgs = read_rosbag(bagname, ["/desired_trajectory_thing"])
        start_time = traj_msgs[0].header.stamp.to_sec() - 5
    else:
        start_time = ball_msgs[0].start_time

    base_traj_msg = read_rosbag(bagname, ["/base_states"], start_time=start_time, duration=50)
    base_states = []
    for msg in base_traj_msg:
        state = [msg.header.stamp.to_sec()]
        state += [pos for pos in msg.position]
        base_states.append(state)
    base_states = np.array(base_states)

    ur10_traj_msg = read_rosbag(bagname, ["/ur10_joint_states"], start_time=start_time, duration=50)
    ur10_states = []
    for msg in ur10_traj_msg:
        state = [msg.header.stamp.to_sec()]
        state += [msg.position[2], msg.position[1], msg.position[0], msg.position[3], msg.position[4], msg.position[5]]
        ur10_states.append(state)
    ur10_states = np.array(ur10_states)
    if (len(ur10_states) == 0):
        print("rosbag %s is wrong" % bagname)
        raise RuntimeError
    ur10_states[:, 0] = ur10_states[:, 0] - start_time
    base_states[:, 0] = base_states[:, 0] - start_time

    base_fit_x = interp1d(base_states[:, 0], base_states[:, 1], kind='linear')
    base_fit_y = interp1d(base_states[:, 0], base_states[:, 2], kind='linear')
    base_fit_r = interp1d(base_states[:, 0], base_states[:, 3], kind='linear')
    thing_traj = []

    for i in range(len(ur10_states)):

        t = ur10_states[i, 0]
        if t > min(base_states[:, 0]) and t < max(base_states[:, 0]):
            point = ur10_states[i, :].tolist() + [base_fit_x(t), base_fit_y(t), base_fit_r(t)]
            point[0] += start_time
            thing_traj.append(point)

    return np.array(thing_traj)


def joint_state_selection(bagname):
    """
    Extracts desired and corresponding actual joint states of the Thing robot from real robot experiments
    :param bagname:
    :return:
        joint_desired_traj:
        [
            [time_from_start_1, joint_0, joint_1, ..., joint_8],
            [...],
            ...
        ]
        joint_actual_states:
        [
            [time_from_start_1, joint_0, joint_1, ..., joint_8],
            [...],
            ...
        ]
    """
    joint_desired_traj = sort_thing_trajectories(bagname)
    start_time = min(joint_desired_traj[:, 0])
    joint_desired_traj[:, 0] -= start_time

    joint_states = sort_joint_states_msg(bagname)
    joint_states[:, 0] -= start_time
    joint_actual_states = []

    # interpolate
    joint_states_interp = []
    for i in range(9):
        joint_states_interp.append(interp1d(joint_states[:, 0], joint_states[:, i + 1], kind='linear'))

    for state in joint_desired_traj:
        t = state[0]
        if t > min(joint_states[:, 0]) and t < max(joint_states[:, 0]):
            joint_state = [t] + [interp(t) for interp in joint_states_interp]
            joint_actual_states.append(joint_state)

    return np.array(joint_desired_traj), np.array(joint_actual_states)


def ros_data_extraction(rosbag_dir, save_path):
    dataset = []
    file_lists = get_file_names(rosbag_dir, ".bag")
    for rosbag in file_lists:
        joint_desired_traj, joint_actual_states = joint_state_selection(rosbag)
        entry = {
            'name': os.path.basename(rosbag),
            'desired_trajectory': joint_desired_traj,
            'actual_trajectory': joint_actual_states
        }
        dataset.append(entry)
        print("Finish processing %s" % entry['name'])

    np.save(save_path, dataset)


if __name__ == "__main__":
    argv = sys.argv
    ros_data_extraction(argv[1], argv[2])

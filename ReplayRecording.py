import pickle
import numpy as np
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
import time
import scipy.interpolate
import matplotlib.pyplot as plt

def get_linspace_steps(data_length: int) -> int:
    num_of_samples = data_length / 19 #19 tracked joints in the robot
    seconds = num_of_samples / 60.0 #Recording sample rate
    return int(seconds / SAMPLE_RATE)

SAMPLE_RATE = 1/60.0 #Sample rate for replaying a recording. NOT the same as input recording sample rate!
TEST = False

with open("Recordings/JointSamples8.pkl", "rb") as file:
    samples = pickle.load(file) #1D Array containing all joint values in order per time step

motion_splines = [] #List of all BSpline objects constructed from input data. Each Spline represents a single joint in the robot
for i in range(19):
    joint_sample = samples[i::19]
    spline = scipy.interpolate.make_splrep(list(range(len(joint_sample))), joint_sample)
    motion_splines.append(spline)

space = np.linspace(0, int(len(samples)/19), get_linspace_steps(len(samples)))
spline_points = []

for spline in motion_splines:
    spline_points.append(spline(space))

if TEST:
    joint_idx: int = 0
    first_joint_data = samples[joint_idx::19]
    first_spline = spline_points[joint_idx]
    plt.plot(list(range(len(first_joint_data))), first_joint_data, label="Original Joint")
    plt.plot(space, first_spline, label="Spline")
    plt.legend()
    plt.show()
    exit(0)

reachy = ReachySDK(host="192.168.68.73")

reachy.turn_off_smoothly("reachy")
reachy.turn_on("reachy")

recorded_joints = [
    reachy.joints.l_shoulder_pitch,
    reachy.joints.l_shoulder_roll,
    reachy.joints.l_arm_yaw,
    reachy.joints.l_elbow_pitch,
    reachy.joints.l_forearm_yaw,
    reachy.joints.l_wrist_pitch,
    reachy.joints.l_wrist_roll,
    reachy.joints.l_gripper,
    reachy.joints.r_shoulder_pitch,
    reachy.joints.r_shoulder_roll,
    reachy.joints.r_arm_yaw,
    reachy.joints.r_elbow_pitch,
    reachy.joints.r_forearm_yaw,
    reachy.joints.r_wrist_pitch,
    reachy.joints.r_wrist_roll,
    reachy.joints.r_gripper,
    reachy.joints.neck_pitch,
    reachy.joints.neck_roll,
    reachy.joints.neck_yaw
]

try:
    first_position = dict(zip(recorded_joints, [l[0] for l in spline_points]))
    goto(first_position, 3.0)

    i = 0 # Tracks sample index
    while i < len(spline_points[0]):
        for j in range(len(recorded_joints)):
            joint = recorded_joints[j]
            joint.goal_position = spline_points[j][i]
        i += 1
        print(f"{i}/{len(spline_points[0])}", end="\r")
        time.sleep(SAMPLE_RATE)
except KeyboardInterrupt:
    pass
finally:
    reachy.turn_off_smoothly("reachy")
    reachy.turn_off("reachy")
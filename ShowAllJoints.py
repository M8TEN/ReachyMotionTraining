import matplotlib.pyplot as plt
import pickle
import scipy.interpolate
import numpy as np

with open("Recordings/JointSamples5.pkl", "rb") as file:
    all_samples = pickle.load(file)

joint_names = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
    "l_gripper",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_arm_yaw",
    "r_elbow_pitch",
    "r_forearm_yaw",
    "r_wrist_pitch",
    "r_wrist_roll",
    "r_gripper",
    "neck_pitch",
    "neck_roll",
    "neck_yaw"
]

for i in range(len(joint_names)):
    arr = all_samples[i::len(joint_names)]
    plt.plot(list(range(int(len(all_samples)/len(joint_names)))), arr, label=joint_names[i])

plt.legend()
plt.show()
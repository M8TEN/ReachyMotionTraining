import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("PitchSamples.pkl", "rb") as file:
    samples = pickle.load(file)

elbow_pitch_samples = []

l_shoulder_pitch = samples[0::8]
l_shoulder_roll = samples[1::8]
l_arm_yaw = samples[2::8]
l_elbow_pitch = samples[3::8]
l_forearm_yaw = samples[4::8]
l_wrist_pitch = samples[5::8]
l_wrist_roll = samples[6::8]
l_gripper = samples[7::8]

for joint_data in [l_shoulder_pitch, l_shoulder_roll, l_arm_yaw, l_elbow_pitch, l_forearm_yaw, l_wrist_pitch, l_wrist_roll, l_gripper]:
    plt.plot(range(len(joint_data)), joint_data)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import numpy as np

def animate(i):
    up_to = min(i, len(l_shoulder_pitch))
    start_from = 0 if (i < 100) else (i - 100)
    xs = l_shoulder_pitch[start_from:up_to:]
    ys = range(start_from, start_from + len(xs))
    ax1.clear()
    ax1.plot(ys, xs)

with open("Recordings/JointSamples1.pkl", "rb") as file:
    samples = pickle.load(file)

MS_PER_SAMPLE = (1/60.0) * 1_000

l_shoulder_pitch = samples[0::19]
l_shoulder_roll = samples[1::19]
l_arm_yaw = samples[2::19]
l_elbow_pitch = samples[3::19]
l_forearm_yaw = samples[4::19]
l_wrist_pitch = samples[5::19]
l_wrist_roll = samples[6::19]
l_gripper = samples[7::19]

names = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_arm_yaw",
    "l_elbow_pitch",
    "l_forearm_yaw",
    "l_wrist_pitch",
    "l_wrist_roll",
    "l_gripper"
]

all_joints = [l_shoulder_pitch, l_shoulder_roll, l_arm_yaw, l_elbow_pitch, l_forearm_yaw, l_wrist_pitch, l_wrist_roll, l_gripper]

# for i in range(len(all_joints)):
#     joint_data = all_joints[i]
#     plt.plot(range(len(joint_data)), joint_data, label=names[i])
# plt.plot(range(len(l_shoulder_pitch)), l_shoulder_pitch, label=names[0])

fig = plt.figure()
ax1 = plt.subplot(1,1,1)
ani = animation.FuncAnimation(fig, animate, interval=MS_PER_SAMPLE)
plt.show()
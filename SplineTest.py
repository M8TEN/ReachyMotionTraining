import pickle
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

SHOW_POINTS: bool = False

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

with open("PitchSamples.pkl", "rb") as file:
    sample_data = pickle.load(file)

l_shoulder_pitch = sample_data[::19]
spline = scipy.interpolate.make_splrep(list(range(len(l_shoulder_pitch))), l_shoulder_pitch)
space = np.linspace(0, len(sample_data) / 60, len(sample_data)*1000)
new_points = spline.__call__(space)

plt.plot(space, new_points, label="Fitted Spline")
plt.plot(list(range(len(l_shoulder_pitch))), l_shoulder_pitch, label="Original Data points")
plt.legend()
plt.show()

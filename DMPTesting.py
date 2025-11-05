import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import pickle

SAMPLE_RATE: float = 1/60.0

# Load the data from disk
with open("PitchSamples.pkl", "rb") as file:
    all_samples = pickle.load(file)

l_shoulder_pitch = all_samples[0::19]
spline = scipy.interpolate.make_splrep(list(range(len(l_shoulder_pitch))), l_shoulder_pitch, s=100)
space = np.linspace(0, len(l_shoulder_pitch), len(l_shoulder_pitch)*1000)
new_points = spline(space)
fd = spline.derivative(1)
sd = spline.derivative(2)
first_derivative_points = fd(space)
second_derivative_points = sd(space)

plt.plot(space, new_points, label="Smoothed Trajectory")
plt.plot(space, first_derivative_points, label="First derivative")
plt.plot(space, second_derivative_points, label="Second derivative")
plt.xlabel("Time Step k")
plt.ylabel("Joint angle")
plt.legend()
plt.show()
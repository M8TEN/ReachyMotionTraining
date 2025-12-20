import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import pickle

NEW_SAMPLE_RATE: float = 1/100.0
ORIGINAL_SAMPLE_RATE: float = 1/60.0

# Load recording data
with open("Recordings/JointSamples8.pkl", "rb") as file:
    all_samples = pickle.load(file)

# Define constants needed in DMP
ALPHA_Z: float = 25.0
BETA_Z: float = ALPHA_Z/4.0
ALPHA_V: float = 25.0
BETA_V: float = ALPHA_V/4.0
TAU: float = (len(all_samples)/19.0) * ORIGINAL_SAMPLE_RATE

def canonical_system(x_cur, v_cur, goal_state, y_cur) -> tuple:
    x_prime = v_cur / TAU
    v_prime = (ALPHA_V * (BETA_V * (goal_state - y_cur) - v_cur)) / TAU
    x_new = x_cur + x_prime * ORIGINAL_SAMPLE_RATE
    v_new = v_cur + v_prime * ORIGINAL_SAMPLE_RATE

    return x_new, v_new

def calculate_f_target(y_demo, y_dot, y_ddot, r):
    f_target = TAU**2 * y_ddot - ALPHA_Z * (BETA_Z * (r - y_demo) - TAU * y_dot)
    return f_target

# Create B-Spline for Joint
l_shoulder_pitch_samples = all_samples[::19]
# Scale time from steps to seconds
time_steps = np.linspace(0, TAU, len(l_shoulder_pitch_samples))
l_shoulder_spline = scipy.interpolate.make_splrep(time_steps, l_shoulder_pitch_samples, s=100)
first_derivative = l_shoulder_spline.derivative()
second_derivative = first_derivative.derivative()

smoothed_samples = l_shoulder_spline(time_steps)
y_dot_arr = first_derivative(time_steps)
y_ddot_arr = second_derivative(time_steps)

# Define state variables
x = y_demo = smoothed_samples[0]
v: float = 0.0
g: float = smoothed_samples[-1]
canonical_matrix = []

f_target_matrix = []
r: float = smoothed_samples[0]
ALPHA_G: float = ALPHA_Z/2.0
ALPHA_X: float = 4.6
s: float = 1.0
phase_matrix = []

# Calculate Phase and f_target matricies
for time_idx in range(len(smoothed_samples)):
    y_demo = smoothed_samples[time_idx]
    y_dot = y_dot_arr[time_idx]
    y_ddot = y_ddot_arr[time_idx]
    x, v = canonical_system(x, v, g, y_demo)
    canonical_matrix.append([x, v])
    f_target = calculate_f_target(y_demo, y_dot, y_ddot, r)
    f_target_matrix.append(f_target)
    r_prime: float = (ALPHA_G * (g-r)) / TAU
    r = r + r_prime * ORIGINAL_SAMPLE_RATE
    s_prime = (-ALPHA_X * s) / TAU
    s += s_prime * ORIGINAL_SAMPLE_RATE
    phase_matrix.append(s)

canonical_matrix = np.array(canonical_matrix)
f_target_matrix = np.array(f_target_matrix)
phase_matrix = np.array(phase_matrix)

# Calculate LWPR
from LWR_Learning import LWR

def calculate_basis(h, c, x):
    return np.exp(-h * (x-c)**2)

lwr = LWR()
num_of_kernels = 50
learned_weights = lwr.learn_weights(f_target_matrix, phase_matrix, num_of_kernels)
centers = np.linspace(1, 0, num_of_kernels)
widths = num_of_kernels**2 / (centers**2 + 1e-10)

plot_x_space = np.linspace(1, 0.0, 10_000)

fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
axs[0].set_title("Unweighted Basis Functions")
axs[1].set_title("Weighted Basis Functions")
for i in range(num_of_kernels):
    basis = calculate_basis(widths[i], centers[i], plot_x_space)
    axs[0].plot(plot_x_space, basis)
    axs[1].plot(plot_x_space, basis * learned_weights[i])
plt.gca().invert_xaxis()
plt.show()
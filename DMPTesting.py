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

canonical_matrix = np.array(canonical_matrix)
f_target_matrix = np.array(f_target_matrix)

# Calculate LWPR

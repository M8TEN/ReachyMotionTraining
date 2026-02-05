import matplotlib.pyplot as plt
import os
import pickle
from DMP import DMP, ALPHA_X, joint_names
from numpy import linspace
from math import sqrt

def rmse(original, predicted) -> float:
    assert(len(original) == len(predicted))
    n = len(original)
    s = sum([(original[i] - predicted[i])**2 for i in range(n)])
    return sqrt(s/n)

try:
    file_num: int = int(input("Which sample? "))
except ValueError:
    print("Could not convert input to integer")
    exit(1)

full_path: str = f"Motions/Motion{file_num}.pkl"

if not os.path.exists(full_path):
    print(f"Could not find file '{full_path}'")
    exit(1)

with open(full_path, "rb") as file:
    all_dmps = pickle.load(file)

ROWS = 5
COLUMNS = 5
errors = [0]*len(all_dmps)

for i in range(len(all_dmps)):
    dmp: DMP = all_dmps[i]
    plt.subplot(ROWS, COLUMNS, i+1)
    time_steps: int = int(dmp.tau / dmp.sample_rate)
    phase = DMP.create_phase_vector(ALPHA_X, dmp.tau, dmp.sample_rate, time_steps)
    recreated_path = dmp.produce_movement(dmp.smooth_samples[0], dmp.smooth_samples[-1], dmp.tau, phase)
    errors[i] = rmse(dmp.smooth_samples, recreated_path)
    tau_space = linspace(0, dmp.tau, time_steps)
    plt.plot(tau_space, dmp.smooth_samples)
    plt.plot(tau_space, recreated_path)
    plt.xlabel("TAU")
    plt.ylabel("Joint Angle (deg)")
    plt.title(joint_names[i])

print(f"Highest deviation: {max(errors)}°")
plt.subplot(ROWS, COLUMNS, len(all_dmps)+1)
plt.bar([str(i) for i in range(len(all_dmps))], errors)
plt.show()
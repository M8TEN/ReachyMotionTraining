import matplotlib.pyplot as plt
import os
import pickle
from DMP import DMP, ALPHA_X, joint_names
from numpy import linspace
from math import sqrt
from reachy_sdk import ReachySDK

reachy_ip: str = "192.168.1.89"

def dist(a, b):
    return sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)

# try:
#     file_num: int = int(input("Which sample? "))
# except ValueError:
#     print("Could not convert input to integer")
#     exit(1)

# full_path: str = f"Motions/Motion{file_num}.pkl"

# if not os.path.exists(full_path):
#     print(f"Could not find file '{full_path}'")
#     exit(1)

for file_path in os.listdir("Motions"):

    full_path: str = os.path.join("Motions", file_path)
    with open(full_path, "rb") as file:
        all_dmps = pickle.load(file)

    print(f"Processing {full_path}")
    originals: list = []
    paths: list = []

    t_steps: int = 0

    for i in range(7):
        dmp: DMP = all_dmps[i]
        t_steps = time_steps = int(dmp.tau / dmp.sample_rate)
        phase = DMP.create_phase_vector(ALPHA_X, dmp.tau, dmp.sample_rate, time_steps)
        originals.append(dmp.smooth_samples)
        paths.append(dmp.produce_movement(dmp.smooth_samples[0], dmp.smooth_samples[-1], dmp.tau, phase))

    reachy: ReachySDK = ReachySDK(host=reachy_ip)

    left_square_sum: float = 0.0

    for i in range(t_steps):
        original_pose = [p[i] for p in originals]
        new_pose = [p[i] for p in paths]
        original_matrix = reachy.l_arm.forward_kinematics(original_pose)
        original_coord = (original_matrix[0][3], original_matrix[1][3], original_matrix[2][3])
        new_matrix = reachy.l_arm.forward_kinematics(new_pose)
        new_coord = (new_matrix[0][3], new_matrix[1][3], new_matrix[2][3])
        d = dist(original_coord, new_coord)
        left_square_sum += d*d

    error = sqrt(left_square_sum/t_steps)
    print(f"{file_path}:\nLeft Arm Error: {error}")
from DMP import DMP, create_phase_vector, joint_names
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

sample_to_load: int = 1
joint_to_display: int = 0

if len(sys.argv) > 1:
    try:
        sample_to_load = int(sys.argv[1])
    except ValueError as e:
        print(e, "Trying to load sample 1 instead")
        sample_to_load = 1

if len(sys.argv) > 2:
    try:
        joint_to_display: int = joint_names.index(sys.argv[2])
    except ValueError:
        print(f"No joint called '{sys.argv[2]}'")
        exit(1)

FILE_PATH: str = f"Recordings/JointSamples{sample_to_load}.pkl"

try:
    with open(FILE_PATH, "rb") as file:
        all_samples = pickle.load(file)
except FileNotFoundError as e:
    print(e)
    exit(1)

NUMBER_OF_JOINTS: int = 19
SAMPLE_RATE: float = 1/60.0
ALPHA_X: float = 25
ALPHA_Z: float = 25
BETA_Z: float = ALPHA_Z/4.0


joint_samples = all_samples[joint_to_display::NUMBER_OF_JOINTS]
TAU: float = len(joint_samples)*SAMPLE_RATE
phase = create_phase_vector(ALPHA_X, TAU, SAMPLE_RATE, len(joint_samples))
dmp: DMP = DMP(SAMPLE_RATE, TAU, ALPHA_Z, ALPHA_X, joint_samples, phase)
dmp.learn_weights()

FACTOR: float = 2
new_motion_start = joint_samples[0]
new_motion_end = joint_samples[-1]
new_time = create_phase_vector(ALPHA_X, TAU*FACTOR, SAMPLE_RATE, int(len(joint_samples)*FACTOR))
new_motion = dmp.produce_movement(new_motion_start, new_motion_end, TAU*FACTOR, new_time)

plt.subplot(1, 2, 1)
plt.title("Original " + joint_names[joint_to_display])
plt.plot(np.linspace(0, TAU, len(joint_samples)), joint_samples)
plt.xlabel("TAU")
plt.ylabel("Joint Position (in deg)")
plt.subplot(1, 2, 2)
plt.title("Recreated " + joint_names[joint_to_display])
plt.plot(np.linspace(0, TAU*FACTOR, len(new_motion)), new_motion)
plt.xlabel("TAU")
plt.ylabel("Joint Position (in deg)")
plt.show()
from reachy_sdk import ReachySDK
import pickle
import time
import numpy as np
import os

TIME_PER_SAMPLE: float = 1/60.0
samples: np.ndarray = np.array([])

reachy: ReachySDK = ReachySDK(host="192.168.68.64")

# reachy.turn_off_smoothly('reachy')
run = True

try:
    RECORDING_TIME: float = float(input("How many seconds should the recording last? "))
except ValueError:
    print("Value must be a number")
    exit(1)

recorded_joints = [
    reachy.joints.l_shoulder_pitch,
    reachy.joints.l_shoulder_roll,
    reachy.joints.l_arm_yaw,
    reachy.joints.l_elbow_pitch,
    reachy.joints.l_forearm_yaw,
    reachy.joints.l_wrist_pitch,
    reachy.joints.l_wrist_roll,
    reachy.joints.l_gripper,
    reachy.joints.r_shoulder_pitch,
    reachy.joints.r_shoulder_roll,
    reachy.joints.r_arm_yaw,
    reachy.joints.r_elbow_pitch,
    reachy.joints.r_forearm_yaw,
    reachy.joints.r_wrist_pitch,
    reachy.joints.r_wrist_roll,
    reachy.joints.r_gripper,
    reachy.joints.neck_pitch,
    reachy.joints.neck_roll,
    reachy.joints.neck_yaw
]

start_time: float = time.time()
print("Starting Recording")
while run and ((time.time() - start_time) < RECORDING_TIME):
    try:
        arm_values = np.array([j.present_position for j in recorded_joints])
        samples = np.append(samples, arm_values)
        print(f"Left arm values at {arm_values}", end="\r")
        time.sleep(TIME_PER_SAMPLE)
    except KeyboardInterrupt:
        print("\nShutting down")
        reachy.turn_off_smoothly("reachy")
        run = False

#reachy.turn_off_smoothly("reachy")
print("\nRecording over, saving results")
# Check for Folder
if not os.path.exists("./Recordings"):
    os.mkdir("./Recordings")
num_of_files: int = len([f for f in os.listdir("./Recordings") if os.path.isfile(f)])
with open(f"Recordings/JointSamples{num_of_files+1}.pkl", "wb") as file:
    pickle.dump(samples, file)
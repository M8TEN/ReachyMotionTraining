from reachy_sdk import ReachySDK
import pickle
import time
import numpy as np
import os
import winsound

TIME_PER_SAMPLE: float = 1/60.0
samples: np.ndarray = np.array([])

reachy: ReachySDK = ReachySDK(host="192.168.68.73")

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

time.sleep(10.0) # Wait time to settle into teleop
winsound.Beep(1_000, 200) # Make a sound to let the user in VR Headset know that recording started
start_time: float = time.time()
print("Starting Recording")
while run and ((time.time() - start_time) < RECORDING_TIME):
    try:
        arm_values = np.array([j.present_position for j in recorded_joints])
        samples = np.append(samples, arm_values)
        time.sleep(TIME_PER_SAMPLE)
    except KeyboardInterrupt:
        print("\nShutting down")
        reachy.turn_off_smoothly("reachy")
        run = False

#reachy.turn_off_smoothly("reachy")
winsound.Beep(1_500, 200) # Make a sound to let the user in VR Headset know that recording ended
print("\nRecording over, saving results")
# Check for Folder
if not os.path.exists("./Recordings"):
    os.mkdir("./Recordings")
num_of_files: int = len([f for f in os.listdir("./Recordings")])
file_path = f"Recordings/JointSamples{num_of_files+1}.pkl"
with open(file_path, "wb") as file:
    pickle.dump(samples, file)

print(f"Saved recording to file {file_path}")
import pickle
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
import time

SAMPLE_RATE = 1/60.0
TEST = False

with open("Recordings/JointSamples8.pkl", "rb") as file:
    samples = pickle.load(file)


print(f"Number of Samples ({len(samples)}) divisible by 19? {len(samples)%19 == 0}")
reachy = ReachySDK(host="192.168.68.73")

#reachy.turn_off_smoothly("reachy")
reachy.turn_on("reachy")

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

try:
    first_position = dict(zip(recorded_joints, samples[:len(recorded_joints):]))
    goto(first_position, 3.0)

    i = 0 # Tracks sample index
    while i < len(samples):
        for j in range(len(recorded_joints)):
            joint = recorded_joints[j]
            joint.goal_position = samples[i+j]
        i += len(recorded_joints)
        print(f"{i}/{len(samples)}", end="\r")
        time.sleep(SAMPLE_RATE)
except KeyboardInterrupt:
    pass
finally:
    reachy.turn_off_smoothly("reachy")
    reachy.turn_off("reachy")
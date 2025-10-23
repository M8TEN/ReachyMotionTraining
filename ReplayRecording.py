import matplotlib.pyplot as plt
import pickle
import numpy as np
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
import time

SAMPLE_RATE = 1/60.0

with open("PitchSamples.pkl", "rb") as file:
    samples = pickle.load(file)

reachy = ReachySDK(host="192.168.68.64")

reachy.turn_off_smoothly("reachy")
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
    first_position = dict(zip(recorded_joints, samples[0:20]))
    goto(first_position, 3.0)

    i = 0
    while i < len(samples):
        for j in range(len(recorded_joints)):
            joint = recorded_joints[j]
            joint.goal_position = samples[i+j]
        i += len(recorded_joints)
        time.sleep(SAMPLE_RATE)
        
    reachy.turn_off_smoothly("reachy")
except KeyboardInterrupt:
    reachy.turn_off_smoothly("reachy")
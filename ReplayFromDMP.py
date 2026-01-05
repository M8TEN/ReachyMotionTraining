from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from DMP import DMP, create_phase_vector, joint_names
import pickle
from time import sleep

with open("Recordings/JointSamples8.pkl", "rb") as file:
    all_samples = pickle.load(file)

NUMBER_OF_JOINTS: int = 19
SAMPLE_RATE: float = 1/100.0
ALPHA_X: float = 25
ALPHA_Z: float = 25
BETA_Z: float = ALPHA_Z/4
SAMPLE_RATE = 1/60.0
TAU = len(all_samples)/NUMBER_OF_JOINTS*SAMPLE_RATE
NUMBER_OF_SAMPLES: int = int(len(all_samples)/NUMBER_OF_JOINTS)

FACTOR: float = 1.0

NEW_TAU: float = TAU*FACTOR
NEW_TIME_STEPS: int = int(NUMBER_OF_SAMPLES*FACTOR)

movement_paths: list = []
phase_vector = create_phase_vector(ALPHA_X, TAU, SAMPLE_RATE, NUMBER_OF_SAMPLES)
reproduce_phase = create_phase_vector(ALPHA_X, NEW_TAU, SAMPLE_RATE, NEW_TIME_STEPS)

for i in range(NUMBER_OF_JOINTS):
    print(f"Calculating DMP for joint '{joint_names[i]}'")
    joint_samples = all_samples[i::NUMBER_OF_JOINTS]
    joint_dmp: DMP = DMP(SAMPLE_RATE, TAU, ALPHA_Z, ALPHA_X, joint_samples, phase_vector)
    joint_dmp.learn_weights()
    recreated_path = joint_dmp.produce_movement(joint_samples[0], joint_samples[-1], NEW_TAU, reproduce_phase)
    print(f"{len(recreated_path)} steps in new movement of joint '{joint_names[i]}'")
    movement_paths.append(recreated_path)

print("Connecting to Reachy..")
reachy: ReachySDK = ReachySDK(host="localhost")
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

reachy.turn_on("reachy")

try:
    first_position = dict(zip(recorded_joints, [p[0] for p in movement_paths]))
    goto(first_position, 3.0)

    for i in range(NEW_TIME_STEPS):
        for j in range(NUMBER_OF_JOINTS):
            recorded_joints[j].goal_position = movement_paths[j][i]
        sleep(SAMPLE_RATE)
except KeyboardInterrupt:
    pass
finally:
    reachy.turn_off_smoothly("reachy")
    reachy.turn_off("reachy")
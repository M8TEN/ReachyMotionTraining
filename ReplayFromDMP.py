from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from DMP import DMP, ALPHA_X, SAMPLE_RATE, NUMBER_OF_JOINTS
import pickle
from time import sleep

LEFT_ARM_INDICIES = (0, 6)
RIGHT_ARM_INDICIES = (8, 14)

# Limits as per Reachy 2023 Documentation
joint_limits: list = [
    (-150, 90), # l_shoulder_pitch
    (-10, 180), # l_shoulder_roll
    (-90, 90), # l_arm_yaw
    (-125, 0), # l_elbow_pitch
    (-100, 100), # l_forearm_yaw
    (-45, 45), # l_wrist_pitch
    (-35, 55), # l_wrist_roll
    (-25, 50), # l_gripper
    (-150, 90), # r_shoulder_pitch
    (-180, 10), # r_shoulder_roll
    (-90, 90), # r_arm_yaw
    (-125, 0), # r_elbow_pitch
    (-100, 100), # r_forearm_yaw
    (-45, 45), # r_wrist_pitch
    (-55, 35), # r_wrist_roll
    (-50, 25), # r_gripper
    (-46, 46), # neck_roll
    (-46, 46), # neck_pitch
    (0, 360) # neck_yaw
]

def clamp(min_val: float, max_val: float, to_clamp: float) -> float:
    return min(max_val, max(min_val, to_clamp))

def play_movement(reachy: ReachySDK, file_path: str, new_tau: float, left_points: tuple, right_points: tuple):

    with open(file_path, "rb") as file:
        all_dmps = pickle.load(file)

    movement_paths: list = []

    NEW_TIME_STEPS: int = 0

    for i in range(len(all_dmps)):
        dmp: DMP = all_dmps[i]
        new_time_steps = int(new_tau / dmp.sample_rate)
        phase = DMP.create_phase_vector(ALPHA_X, new_tau, dmp.sample_rate, new_time_steps)

        if (i >= LEFT_ARM_INDICIES[0] and i <= LEFT_ARM_INDICIES[1]):
            motion_start = left_points[0][i] if (left_points[0]) else dmp.original_path[0]
            motion_end = left_points[1][i] if (left_points[1]) else dmp.original_path[-1]
        elif (i >= RIGHT_ARM_INDICIES[0] and i <= RIGHT_ARM_INDICIES[1]):
            motion_start = right_points[0][i-RIGHT_ARM_INDICIES[0]] if (right_points[0]) else dmp.original_path[0]
            motion_end = right_points[1][i-RIGHT_ARM_INDICIES[0]] if (right_points[1]) else dmp.original_path[-1]
        else:
            motion_start = dmp.original_path[0]
            motion_end = dmp.original_path[-1]

        new_movement_path = dmp.produce_movement(motion_start, motion_end, new_tau, phase)
        movement_paths.append(new_movement_path)
        NEW_TIME_STEPS = len(new_movement_path)

    print("Turning on Reachy..")
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
        first_position = dict(zip(recorded_joints, (p[0] for p in movement_paths)))
        goto(first_position, 3.0)
        for i in range(NEW_TIME_STEPS):
            for j in range(NUMBER_OF_JOINTS):
                recorded_joints[j].goal_position = clamp(joint_limits[j][0], joint_limits[j][1], movement_paths[j][i])
            sleep(SAMPLE_RATE)
    except KeyboardInterrupt:
        pass
    finally:
        reachy.turn_off_smoothly("reachy")
        reachy.turn_off("reachy")
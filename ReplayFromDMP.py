from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from DMP import DMP, ALPHA_X, SAMPLE_RATE, NUMBER_OF_JOINTS
import pickle
from time import sleep, time
import numpy as np

LEFT_ARM_INDICIES = (0, 6)
RIGHT_ARM_INDICIES = (8, 14)

LEFT_SIDE: int = 0
RIGHT_SIDE: int = 1

LEFT_VOLUME_LIMITS: tuple = (
    0.14888854103409055,  #x-min
    0.3953221660406556,   #x-max
    0.004753014205407358, #y-min
    0.6103183116686676,   #y-max
    -0.39519877482349264, #z-min
    0.5045470562610281    #z-max
)

RIGHT_VOLUME_LIMITS: tuple = (
    0.21701274404719764,  #x-min
    0.6186213994503541,   #x-max
    -0.5567124991918473,  #y-min
    -0.13960405374265367, #y-max
    -0.3996426656448221,  #z-min
    0.5467782688288836    #z-max
)


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

def limit_to_volume(reachy: ReachySDK, arm_paths: np.ndarray, side: int) -> np.ndarray:
    if (side == LEFT_SIDE):
        limits = LEFT_VOLUME_LIMITS
        arm = reachy.l_arm
    elif (side == RIGHT_SIDE):
        limits = RIGHT_VOLUME_LIMITS
        arm = reachy.r_arm
    else:
        print(f"WARNING: Unknown side {side}")
        return arm_paths
    
    for i in range(len(arm_paths[0])): # Iterate over every 3D-Position in movement path
        joint_angles = [p[i] for p in arm_paths]
        pose = arm.forward_kinematics(joint_angles)
        # In Reachy's end effector space, X = Forward, Y = Right, Z = Up
        pose[0][3] = clamp(limits[0], limits[1], pose[0][3]) # X
        pose[1][3] = clamp(limits[2], limits[3], pose[0][3]) # Y
        pose[2][3] = clamp(limits[4], limits[5], pose[0][3]) # Z
        joint_pos = arm.inverse_kinematics(pose)
        for j in range(len(joint_pos)):
            arm_paths[0][j] = joint_pos[j]
            arm_paths[1][j] = joint_pos[j]
            arm_paths[2][j] = joint_pos[j]
            arm_paths[3][j] = joint_pos[j]
            arm_paths[4][j] = joint_pos[j]
            arm_paths[5][j] = joint_pos[j]
            arm_paths[6][j] = joint_pos[j]
    
    return arm_paths


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

    clamped_left_arm_joints = limit_to_volume(reachy, movement_paths[LEFT_ARM_INDICIES[0]:LEFT_ARM_INDICIES[1]+1:], LEFT_SIDE)
    clamped_right_arm_joints = limit_to_volume(reachy, movement_paths[RIGHT_ARM_INDICIES[0]:RIGHT_ARM_INDICIES[1]+1:], RIGHT_SIDE)

    for i in range(len(clamped_left_arm_joints)):
        for j in range(len(clamped_left_arm_joints[i])):
            movement_paths[i][j] = clamped_left_arm_joints[i][j]

    for i in range(len(clamped_right_arm_joints)):
        for j in range(len(clamped_right_arm_joints[i])):
            movement_paths[i][j] = clamped_right_arm_joints[i][j]

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
            start_of_frame: float = time()
            for j in range(NUMBER_OF_JOINTS):
                recorded_joints[j].goal_position = clamp(joint_limits[j][0], joint_limits[j][1], movement_paths[j][i])
            sleep(max(0, SAMPLE_RATE - (time()-start_of_frame)))
    except KeyboardInterrupt:
        pass
    finally:
        reachy.turn_off_smoothly("reachy")
        reachy.turn_off("reachy")

if __name__ == "__main__":
    robot = ReachySDK(host="192.168.1.89")
    play_movement(robot, "Motions/Motion1.pkl", 10.2, (None, None), (None, None))
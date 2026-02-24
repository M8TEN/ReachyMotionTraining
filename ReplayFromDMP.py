from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from DMP import DMP, ALPHA_X, SAMPLE_RATE, NUMBER_OF_JOINTS
import pickle
from time import sleep, time
import numpy as np

LEFT_ARM_INDICIES: tuple = (0, 6)
RIGHT_ARM_INDICIES: tuple = (8, 14)

LEFT_SIDE: int = 0
RIGHT_SIDE: int = 1

LEFT_VOLUME_LIMITS: tuple = (
     0.1,  #x-min
     0.65, #x-max
     0.1,  #y-min
     0.65, #y-max
    -0.35, #z-min
     0.65  #z-max
)

RIGHT_VOLUME_LIMITS: tuple = (
     0.1,  #x-min
     0.65, #x-max
    -0.65, #y-max
    -0.1,  #y-min
    -0.35, #z-min
     0.65  #z-max
)


# Limits as per Reachy 2023 Documentation
JOINT_LIMITS: list = [
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

def validate_motor_positions(joint_positions, start_offset) -> bool:
    for j in range(len(joint_positions)):
            joint_min, joint_max = JOINT_LIMITS[j+start_offset]
            if joint_positions[j] < joint_min or joint_positions[j] > joint_max:
                return False
    
    return True

def limit_to_volume(reachy: ReachySDK, arm_paths: np.ndarray, side: int) -> None:
    if side == LEFT_SIDE:
        arm = reachy.l_arm
        limits = LEFT_VOLUME_LIMITS
        limit_offset = LEFT_ARM_INDICIES[0]
    elif side == RIGHT_SIDE:
        arm = reachy.r_arm
        limits = RIGHT_VOLUME_LIMITS
        limit_offset = RIGHT_ARM_INDICIES[0]

    last_pos = [p[0] for p in arm_paths]
    for i in range(len(last_pos)):
        last_pos[i] = clamp(JOINT_LIMITS[i+limit_offset][0], JOINT_LIMITS[i+limit_offset][1], last_pos[i])
    last_pose = arm.forward_kinematics(last_pos)

    for i in range(1, len(arm_paths[0])):
        current_positions = [p[i] for p in arm_paths]
        current_pose = arm.forward_kinematics(current_positions)

        is_inside: bool = (
            (current_pose[0][3] > limits[0] and current_pose[0][3] < limits[1]) and
            (current_pose[1][3] > limits[2] and current_pose[1][3] < limits[3]) and
            (current_pose[2][3] > limits[4] and current_pose[2][3] < limits[5])
        )

        if is_inside:
            if validate_motor_positions(current_positions, limit_offset):
                last_pos = current_positions
                last_pose = current_pose
            else:
                for j in range(len(arm_paths)):
                    arm_paths[j][i] = last_pos[j]
            continue

        direction = (
            current_pose[0][3] - last_pose[0][3],
            current_pose[1][3] - last_pose[1][3],
            current_pose[2][3] - last_pose[2][3],
        )
        start = (
            last_pose[0][3],
            last_pose[1][3],
            last_pose[2][3]
        )

        all_t_values = []
        if direction[0] < -1e-6:
            all_t_values.append((limits[0] - start[0]) / direction[0])
        if direction[0] > 1e-6: 
            all_t_values.append((limits[1] - start[0]) / direction[0])
        if direction[1] < -1e-6:
            all_t_values.append((limits[2] - start[1]) / direction[1])
        if direction[1] > 1e-6:
            all_t_values.append((limits[3] - start[1]) / direction[1])
        if direction[2] < -1e-6:
            all_t_values.append((limits[4] - start[2]) / direction[2])
        if abs(direction[2]) > 1e-6:
            all_t_values.append((limits[5] - start[2]) / direction[2])

        filtered_t_values = [t for t in all_t_values if (t >= -1e-6)]
        if (len(filtered_t_values) == 0):
            for j in range(len(arm_paths)):
                arm_paths[j][i] = last_pos[j]
            continue
        
        smallest_t = max(0.0, min(filtered_t_values))

        if smallest_t > 1.0:
            if validate_motor_positions(current_positions, limit_offset):
                last_pos = current_positions
                last_pose = current_pose
                continue
            else:
                smallest_t = 1.0

        ik_success: bool = True
        current_pose[0][3] = start[0] + direction[0] * smallest_t
        current_pose[1][3] = start[1] + direction[1] * smallest_t
        current_pose[2][3] = start[2] + direction[2] * smallest_t
        try:
            current_positions = arm.inverse_kinematics(current_pose)
        except ValueError:
            ik_success = False
        
        is_valid = ik_success and validate_motor_positions(current_positions, limit_offset)

        if is_valid:
            for j in range(len(arm_paths)):
                arm_paths[j][i] = current_positions[j]
            last_pos = current_positions
            last_pose = current_pose
        else:
            for j in range(len(arm_paths)):
                arm_paths[j][i] = last_pos[j]

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

    limit_to_volume(reachy, movement_paths[LEFT_ARM_INDICIES[0]:LEFT_ARM_INDICIES[1]+1:], LEFT_SIDE)
    limit_to_volume(reachy, movement_paths[RIGHT_ARM_INDICIES[0]:RIGHT_ARM_INDICIES[1]+1:], RIGHT_SIDE)

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
                recorded_joints[j].goal_position = movement_paths[j][i]
            sleep(max(0, SAMPLE_RATE - (time()-start_of_frame)))
    except KeyboardInterrupt:
        pass
    finally:
        reachy.turn_off_smoothly("reachy")
        reachy.turn_off("reachy")

if __name__ == "__main__":
    robot = ReachySDK(host="192.168.1.89")
    play_movement(robot, "Motions/Motion1.pkl", 10.2, (None, None), (None, None))
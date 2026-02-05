from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from DMP import DMP, ALPHA_X, SAMPLE_RATE, NUMBER_OF_JOINTS
import pickle
from time import sleep

joint_limits: dict = {
    "r_shoulder_pitch": (-150, 90),
    "r_shoulder_roll": (-180, 10),
    "r_arm_yaw": (-90, 90),
    "r_elbow_pitch": (-125, 0),
    "r_forearm_yaw": (-100, 100),
    "r_wrist_pitch": (-45, 45),
    "r_wrist_roll": (-55, 35),
    "r_gripper": (-50, 25),
    "l_shoulder_pitch": (-150, 90),
    "l_shoulder_roll": (-10, 180),
    "l_arm_yaw": (-90, 90),
    "l_elbow_pitch": (-125, 0),
    "l_forearm_yaw": (-100, 100),
    "l_wrist_pitch": (-45, 45),
    "wrist_roll": (-35, 55),
    "l_gripper": (-25, 50),
    "neck_roll": (-46, 46),
    "neck_pitch": (-46, 46),
    "neck_yaw": (0, 360)
}

def play_movement(reachy: ReachySDK, file_path: str, new_tau: float):

    with open(file_path, "rb") as file:
        all_dmps = pickle.load(file)

    movement_paths: list = []

    NEW_TIME_STEPS: int = 0

    for i in range(len(all_dmps)):
        dmp: DMP = all_dmps[i]
        new_time_steps = int(new_tau / dmp.sample_rate)
        phase = DMP.create_phase_vector(ALPHA_X, new_tau, SAMPLE_RATE, new_time_steps)
        new_movement_path = dmp.produce_movement(dmp.smooth_samples[0], dmp.smooth_samples[-1], new_tau, phase)
        movement_paths.append(new_movement_path)
        NEW_TIME_STEPS = len(new_movement_path)

    print("Connecting to Reachy..")
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
        print(reachy.r_arm.forward_kinematics())
        for i in range(NEW_TIME_STEPS):
            for j in range(NUMBER_OF_JOINTS):
                recorded_joints[j].goal_position = movement_paths[j][i]
            sleep(SAMPLE_RATE)
    except KeyboardInterrupt:
        pass
    finally:
        reachy.turn_off_smoothly("reachy")
        reachy.turn_off("reachy")
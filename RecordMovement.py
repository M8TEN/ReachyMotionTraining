from reachy_sdk import ReachySDK
import pickle
import time
import numpy as np
import os

class MotionRecorder():
    def __init__(self, sample_rate: float = 1/60.0, robot_ip: str = "localhost") -> None:
        self.sample_rate: float = sample_rate
        self.samples: list = []
        self.recording: bool = False
        self.start_time: float = time.time()
        self.reachy: ReachySDK = ReachySDK(host=robot_ip)
        self.recorded_joints: list = [
            self.reachy.joints.l_shoulder_pitch,
            self.reachy.joints.l_shoulder_roll,
            self.reachy.joints.l_arm_yaw,
            self.reachy.joints.l_elbow_pitch,
            self.reachy.joints.l_forearm_yaw,
            self.reachy.joints.l_wrist_pitch,
            self.reachy.joints.l_wrist_roll,
            self.reachy.joints.l_gripper,
            self.reachy.joints.r_shoulder_pitch,
            self.reachy.joints.r_shoulder_roll,
            self.reachy.joints.r_arm_yaw,
            self.reachy.joints.r_elbow_pitch,
            self.reachy.joints.r_forearm_yaw,
            self.reachy.joints.r_wrist_pitch,
            self.reachy.joints.r_wrist_roll,
            self.reachy.joints.r_gripper,
            self.reachy.joints.neck_pitch,
            self.reachy.joints.neck_roll,
            self.reachy.joints.neck_yaw
        ]
    
    def update(self):
        if self.recording:
            joint_values: list = [j.present_position for j in self.recorded_joints]
            self.samples += joint_values

    def find_higest_num(self, dir: str) -> int:
        highest_num: int = 1
        all_file_names: list = [f for f in os.listdir(dir) if os.path.isfile(f)]
        for file_name in all_file_names:
            dot_idx: int = file_name.rfind(".")-1
            if (dot_idx == -1): dot_idx = len(file_name)-1
            num_str: str = ""
            while (file_name[dot_idx] >= 0 and file_name[dot_idx].isdigit()):
                num_str = file_name[dot_idx] + num_str
                dot_idx -= 1
            highest_num = max(highest_num, int(num_str))
        
        return highest_num

    def start_recording(self) -> None:
        self.samples.clear()
        self.start_time = time.time()
        self.recording = True
    
    def stop_recording(self) -> None:
        self.recording = False
        print(f"Recorded for {time.time() - self.start_time} seconds")
        self.save_recording()
    
    def save_recording(self, dir: str = "Recordings") -> None:
        if len(self.samples) == 0: return
        if not os.path.exists(dir):
            os.mkdir(dir)
        movement_num: int = self.find_higest_num(dir)+1
        new_file_path: str = f"{dir}/JointSamples{movement_num}.pkl"
        with open(new_file_path, "wb") as file:
            pickle.dump(np.array(self.samples), file)
        print(f"Saved recording to '{new_file_path}'")

if __name__ == "__main__":
    try:
        time_to_record = float(input("How long should the recording be? "))
    except TypeError:
        print("Input must be int or float")
        exit(1)
    
    recorder = MotionRecorder(1/100.0)
    WAIT_DELAY: float = 15.0
    time.sleep(WAIT_DELAY)
    recorded_time: float = 0.0
    print("\a")
    recorder.start_recording()
    while recorded_time < time_to_record:
        recorder.update()
        time.sleep(recorder.sample_rate)
        recorded_time += recorder.sample_rate
    
    recorder.stop_recording()
    print("\a")
    print("\a")
    print("Recording finished")
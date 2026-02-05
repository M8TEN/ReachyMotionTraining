from reachy_sdk import ReachySDK
import pickle
import time
import numpy as np
import os
import asyncio
from DMP import DMP, ALPHA_X, ALPHA_Z
from multiprocessing import Pool
import asyncio

RECORDING_START: int = 1
RECORDING_END: int = 2
NO_REQUEST: int = 3
ALLOW_REQUESTS: int = 4
NEEDED_LENGTH: int = 2
DUMMY: int = 0xFF

class MotionRecorder():
    def __init__(self, sample_rate: float = 1/60.0, robot_ip: str = "localhost") -> None:
        self.sample_rate: float = sample_rate
        self.samples: list = []
        self.recording: bool = False
        self.start_time: float = time.time()
        self.pool = Pool()
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
    
    def __del__(self):
        self.pool.close()
        self.pool.join()
    
    async def update(self, halt_event: asyncio.Event):
        while not halt_event.is_set():
            if self.recording:
                joint_values: list = [j.present_position for j in self.recorded_joints]
                self.samples += joint_values
            await asyncio.sleep(self.sample_rate)

    def find_higest_num(self, dir: str) -> int:
        highest_num: int = 0
        all_file_names: list = [f for f in os.listdir(dir) if os.path.isfile(dir+"/"+f)]
        for file_name in all_file_names:
            dot_idx: int = file_name.rfind(".")-1
            if (dot_idx == -1): dot_idx = len(file_name)-1
            num_str: str = ""
            while (dot_idx >= 0 and file_name[dot_idx].isdigit()):
                num_str = file_name[dot_idx] + num_str
                dot_idx -= 1
            highest_num = max(highest_num, int(num_str))
        
        return highest_num

    async def start_recording(self, writer: asyncio.StreamWriter, start_event: asyncio.Event, halt_event: asyncio.Event) -> None:
        while not halt_event.is_set():
            await start_event.wait()
            start_event.clear()
            print("Starting recording")
            self.samples.clear()
            await self.send_client_command(writer, bytes([RECORDING_START, NO_REQUEST]))
            self.start_time = time.time()
            self.recording = True
    
    async def stop_recording(self, writer: asyncio.StreamWriter, stop_event: asyncio.Event, halt_event: asyncio.Event) -> None:
        while not halt_event.is_set():
            await stop_event.wait()
            stop_event.clear()
            self.recording = False
            recording_time: float = time.time() - self.start_time
            print(f"Recorded for {recording_time} seconds")
            await self.send_client_command(writer, bytes([RECORDING_END, NO_REQUEST]))
            if len(self.samples) > 0:
                dmps = await self.calculate_dmps()
                print("Saving recording..")
                self.save_recording(dmps)
            await self.send_client_command(writer, bytes([ALLOW_REQUESTS]))
    
    @staticmethod
    def dmp_helper(tau: float, sample_rate: float, samples) -> DMP:
        phase = DMP.create_phase_vector(ALPHA_X, tau, sample_rate, len(samples))
        dmp = DMP(sample_rate, tau, ALPHA_Z, ALPHA_X, samples, phase)
        dmp.learn_weights()
        return dmp

    def run_starmap(self, grouped_samples):
        return self.pool.starmap(MotionRecorder.dmp_helper, grouped_samples)

    async def calculate_dmps(self) -> list:
        np_samples: np.ndarray = np.array(self.samples)
        number_of_samples: int = len(np_samples)
        number_of_joints: int = len(self.recorded_joints)
        tau: float = (number_of_samples / number_of_joints) * self.sample_rate
        grouped_samples = [(tau, self.sample_rate, np_samples[i::number_of_joints]) for i in range(number_of_joints)]
        
        dmps = await asyncio.get_running_loop().run_in_executor(None, self.run_starmap, grouped_samples)

        return dmps

    def save_recording(self, dmps: list, dir: str = "Motions") -> None:
        if len(dmps) == 0: return
        if not os.path.exists(dir):
            os.mkdir(dir)
        movement_num: int = self.find_higest_num(dir)+1
        new_file_path: str = f"{dir}/Motion{movement_num}.pkl"
        with open(new_file_path, "wb") as file:
            pickle.dump(np.array(dmps), file)
        print(f"Saved recording to '{new_file_path}'")
    
    async def send_client_command(self, writer: asyncio.StreamWriter, command: bytes):
        if len(command) < NEEDED_LENGTH:
            command += bytes([DUMMY]*(NEEDED_LENGTH - len(command)))
        writer.write(command)
        await writer.drain()
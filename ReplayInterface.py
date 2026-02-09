from tkinter import *
from tkinter import ttk
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from DMP import DMP, joint_names
from ReplayFromDMP import play_movement
from reachy_sdk import ReachySDK

def find_num(file_name: str) -> int:
    dot_idx: int = file_name.rfind(".")
    num: str = ""
    idx: int = dot_idx-1
    while (idx >= 0 and file_name[idx].isdigit()):
        num = file_name[idx] + num
        idx -= 1
    
    return int(num)

def show_selected_sample():
    global loaded_sample
    print(loaded_sample)
    ROWS = COLUMNS = 5
    number_of_samples: int = len(loaded_sample) // 19
    total_time: float = number_of_samples * 1/60.0
    time_axis: np.ndarray = np.linspace(0, total_time, number_of_samples)
    for i in range(len(joint_names)):
        joint_samples = loaded_sample[i::19]
        plt.subplot(ROWS, COLUMNS, i+1)
        plt.plot(time_axis, joint_samples, label=joint_names[i])
    
    plt.show()

def select_sample(path: str) -> None:
    global loaded_sample, loaded_file_path
    with open(path, "rb") as file:
        loaded_sample = pickle.load(file)
    
    first_dmp: DMP = loaded_sample[0]
    time: float = first_dmp.tau
    slider.configure(from_=time/2.0, to=time*2.0)
    slider.set(time)
    play_button.configure(state="enabled")
    loaded_file_path = path
    reset_motion()
    selection_label.configure(text="Selected Motion " + str(find_num(path)))

def set_left_start() -> None:
    global l_motion_start
    l_motion_start = [
        reachy.joints.l_shoulder_pitch.present_position,
        reachy.joints.l_shoulder_roll.present_position,
        reachy.joints.l_arm_yaw.present_position,
        reachy.joints.l_elbow_pitch.present_position,
        reachy.joints.l_forearm_yaw.present_position,
        reachy.joints.l_wrist_pitch.present_position,
        reachy.joints.l_wrist_roll.present_position
    ]

def set_right_start() -> None:
    global r_motion_start
    r_motion_start = [
        reachy.joints.r_shoulder_pitch.present_position,
        reachy.joints.r_shoulder_roll.present_position,
        reachy.joints.r_arm_yaw.present_position,
        reachy.joints.r_elbow_pitch.present_position,
        reachy.joints.r_forearm_yaw.present_position,
        reachy.joints.r_wrist_pitch.present_position,
        reachy.joints.r_wrist_roll.present_position
    ]

def set_left_end() -> None:
    global l_motion_end
    l_motion_end = [
        reachy.joints.l_shoulder_pitch.present_position,
        reachy.joints.l_shoulder_roll.present_position,
        reachy.joints.l_arm_yaw.present_position,
        reachy.joints.l_elbow_pitch.present_position,
        reachy.joints.l_forearm_yaw.present_position,
        reachy.joints.l_wrist_pitch.present_position,
        reachy.joints.l_wrist_roll.present_position
    ]

def set_right_end() -> None:
    global r_motion_end
    r_motion_end = [
        reachy.joints.r_shoulder_pitch.present_position,
        reachy.joints.r_shoulder_roll.present_position,
        reachy.joints.r_arm_yaw.present_position,
        reachy.joints.r_elbow_pitch.present_position,
        reachy.joints.r_forearm_yaw.present_position,
        reachy.joints.r_wrist_pitch.present_position,
        reachy.joints.r_wrist_roll.present_position
    ]

def replay_on_robot() -> None:
    global reachy, loaded_file_path, slider, l_motion_start, l_motion_end, r_motion_start, r_motion_end
    log_positions("DebugLog.txt")
    play_movement(reachy, loaded_file_path, slider.get(), (l_motion_start, l_motion_end), (r_motion_start, r_motion_end))

def reset_motion() -> None:
    global loaded_sample, slider, l_motion_start, l_motion_end, r_motion_start, r_motion_end
    l_motion_start = l_motion_end = r_motion_start = r_motion_end = None
    if len(loaded_sample) > 0:
        slider.set(loaded_sample[0].tau)

def log_positions(file_path: str) -> None:
    message: str = f"l_motion_start: {l_motion_start}\nl_motion_end: {l_motion_end}\nr_motion_start: {r_motion_start}\nr_motion_end{r_motion_end}"
    with open(file_path, "w") as file:
        file.write(message)

reachy: ReachySDK = ReachySDK(host="127.0.0.1")
loaded_sample = np.array([])
loaded_file_path: str = ""
l_motion_start: list = None
l_motion_end: list = None
r_motion_start: list = None
r_motion_end: list = None

root = Tk()
root.title("Reachy Motion Replay")

mainframe: Frame = Frame(root, width=500, height=500)
mainframe.grid(column=0, row=0, sticky=(N, E, S, W))

selection_label: ttk.Label = ttk.Label(mainframe, text="No Motion selected")
selection_label.grid(row=0, column=1, sticky=(E,W))

slider: Scale = Scale(mainframe, orient="horizontal", resolution=0.1, label="Execution time (s)")
slider.grid(column=1, row=1, sticky=(E,W))

button_frame: Frame = Frame(mainframe)
button_frame.grid(column=0, row=0, rowspan=3, sticky=(N, S))

text: Text = Text(button_frame, width=15, height=20)

sb: ttk.Scrollbar = ttk.Scrollbar(button_frame, orient="vertical", command=text.yview)
sb.pack(side="right", fill="y")

text.configure(yscrollcommand=sb.set, state="disabled")

file_names: list = os.listdir("Motions")
file_names.sort(key=find_num)

for f in file_names:
    full_path: str = "Motions/" + f
    btn = ttk.Button(text, text=f.removesuffix(".pkl"), command=lambda msg=full_path: select_sample(msg))
    text.window_create("end", window=btn)
    text.insert("end", "\n")

text.pack()

command_button_frame: ttk.Frame = ttk.Frame(mainframe)
left_start_button: ttk.Button = ttk.Button(command_button_frame, text="Set Left Start", command=set_left_start)
left_end_button: ttk.Button = ttk.Button(command_button_frame, text="Set Left End", command=set_left_end)
right_start_button: ttk.Button = ttk.Button(command_button_frame, text="Set Right Start", command=set_right_start)
right_end_button: ttk.Button = ttk.Button(command_button_frame, text="Set Right End", command=set_right_end)
play_button: ttk.Button = ttk.Button(command_button_frame, text="Replay Motion", command=replay_on_robot)
reset_button: ttk.Button = ttk.Button(command_button_frame, text="Reset Motion Parameters", command=reset_motion)
command_button_frame.grid(row=2, column=1)
left_start_button.grid(row=0, column=0)
left_end_button.grid(row=1, column=0)
right_start_button.grid(row=0, column=1)
right_end_button.grid(row=1, column=1)
reset_button.grid(row=0, column=3, padx=20)
play_button.grid(row=1, column=3, padx=20)
play_button.configure(state="disabled")

root.mainloop()
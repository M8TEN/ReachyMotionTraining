from tkinter import *
from tkinter import ttk
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from DMP import DMP, joint_names
from ReplayFromDMP import play_movement

loaded_sample = np.array([])
loaded_file_path: str = ""

def show_selected_sample():
    global loaded_sample
    print(loaded_sample)
    ROWS = COLUMNS = 5
    number_of_samples = len(loaded_sample) // 19
    total_time: float = number_of_samples * 1/60.0
    time_axis = np.linspace(0, total_time, number_of_samples)
    for i in range(len(joint_names)):
        joint_samples = loaded_sample[i::19]
        plt.subplot(ROWS, COLUMNS, i+1)
        plt.plot(time_axis, joint_samples, label=joint_names[i])
    
    plt.show()

def test(path: str):
    global loaded_sample, loaded_file_path
    with open(path, "rb") as file:
        loaded_sample = pickle.load(file)
    
    first_dmp: DMP = loaded_sample[0]
    time = first_dmp.tau
    slider.configure(from_=time/2.0, to=time*2.0)
    slider.set(time)
    start_button.configure(state="enabled")
    loaded_file_path = path



root = Tk()
root.title("Reachy Motion Replay")

mainframe = Frame(root, width=500, height=500)
mainframe.grid(column=0, row=0, sticky=(N, E, S, W))

slider = Scale(mainframe, orient="horizontal", resolution=0.1, label="Execution time (s)")
slider.grid(column=1, row=0, sticky=(E,W))

button_frame = Frame(mainframe)
button_frame.grid(column=0, row=0, sticky=(N, S))

text = Text(button_frame)

sb = ttk.Scrollbar(button_frame, orient="vertical", command=text.yview)
sb.pack(side="right", fill="y")

text.configure(yscrollcommand=sb.set)

for f in os.listdir("Motions"):
    full_path: str = "Motions/" + f
    btn = ttk.Button(text, text=f, command=lambda msg=full_path: test(msg))
    text.window_create("end", window=btn)
    text.insert("end", "\n")

text.pack()

start_button = ttk.Button(mainframe, text="Replay Motion", command=lambda: play_movement(loaded_file_path, slider.get()))
start_button.grid(row=1, column=1)
start_button.configure(state="disabled")

root.mainloop()
from DMP import DMP, joint_names
import pickle
import os

NUMBER_OF_JOINTS: int = 19
SAMPLE_RATE: float = 1/60.0
ALPHA_X = 25
ALPHA_Z = 50
dir_path: str = "Recordings"
target_path: str = "Motions"

if not os.path.exists(target_path):
    os.mkdir(target_path)

for file_name in os.listdir(dir_path):
    with open(dir_path+"/"+file_name, "rb") as file:
        all_samples = pickle.load(file)
    
    dmps: list = []
    for i in range(NUMBER_OF_JOINTS):
        print(f"Calculating DMP for joint {joint_names[i]}")
        joint_samples = all_samples[i::NUMBER_OF_JOINTS]
        tau: float = len(joint_samples) * SAMPLE_RATE
        phase = DMP.create_phase_vector(ALPHA_X, tau, SAMPLE_RATE, len(joint_samples))
        joint_dmp: DMP = DMP(SAMPLE_RATE, tau, ALPHA_Z, ALPHA_X, joint_samples, phase)
        joint_dmp.learn_weights()
        dmps.append(joint_dmp)

    file_number: str = ""
    dot_idx: int = file_name.rfind(".")
    num_idx: int = dot_idx-1
    while num_idx >= 0 and file_name[num_idx].isnumeric():
        file_number = file_name[num_idx] + file_number
        num_idx -= 1
    
    full_path: str = f"{target_path}/Motion{file_number}.pkl"
    with open(full_path, "wb") as out_file:
        pickle.dump(dmps, out_file)
    
    print(f"Saved File {full_path}")
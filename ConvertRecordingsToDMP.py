from DMP import DMP, joint_names
import pickle
import os
from multiprocessing import Pool

NUMBER_OF_JOINTS: int = 19
SAMPLE_RATE: float = 1/60.0
ALPHA_X = 25
ALPHA_Z = 50
dir_path: str = "Recordings"
target_path: str = "Motions"

def create_dmp(samples) -> DMP:
    tau: float = len(samples) * SAMPLE_RATE
    phase = DMP.create_phase_vector(ALPHA_X, tau, SAMPLE_RATE, len(samples))
    joint_dmp: DMP = DMP(SAMPLE_RATE, tau, ALPHA_Z, ALPHA_X, samples, phase)
    joint_dmp.learn_weights()
    return joint_dmp

if __name__ == "__main__":

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    with Pool() as pool:
        for file_name in os.listdir(dir_path):
            with open(dir_path+"/"+file_name, "rb") as file:
                all_samples = pickle.load(file)
            
            print(f"Processing file {file_name}")
            ordered_samples = [all_samples[i::NUMBER_OF_JOINTS] for i in range(NUMBER_OF_JOINTS)]
            dmps: list = pool.map(create_dmp, ordered_samples)

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
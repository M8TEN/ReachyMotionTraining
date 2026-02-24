import os
import sys
import pickle
from DMP import DMP, ALPHA_X, joint_names
from math import sqrt

def rmse(original, predicted) -> float:
    try:
        assert(len(original) == len(predicted))
    except AssertionError:
        return 0.0

    n = len(original)
    s = sum([(original[i] - predicted[i])**2 for i in range(n)])
    return sqrt(s/n)

def float_to_xlsx_string(num: float) -> str:
    return str(num).replace(".", ",")

upper_path: str = "MotionArchive/VP"
if len(sys.argv) > 1:
    try:
        dir_num = int(sys.argv[1])
    except ValueError:
        dir_num = 1
else:
    dir_num = 1

dir_path: str = upper_path + str(dir_num)

log_path: str = "RMSE_Log.txt"

if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "a") as log_file:
    while os.path.exists(dir_path):

        for file_name in os.listdir(dir_path):
            full_path: str = dir_path + "/" + file_name
            print("Processing " + full_path)
            log_file.write(full_path + ":\n")
            with open(full_path, "rb") as dmp_file:
                all_dmps = pickle.load(dmp_file)
            errors = [0] * len(all_dmps)
            for i in range(len(all_dmps)):
                print(f"Calculating RMSE for DMP {joint_names[i]}")
                dmp: DMP = all_dmps[i]
                time_steps: int = int(round(dmp.tau / dmp.sample_rate))
                phase = DMP.create_phase_vector(ALPHA_X, dmp.tau, dmp.sample_rate, time_steps)
                start_pos = dmp.original_path[0]
                end_pos = dmp.original_path[-1]
                recreated_path = dmp.produce_movement(start_pos, end_pos, dmp.tau, phase)
                errors[i] = rmse(dmp.original_path, recreated_path)
                log_file.write(f"  {joint_names[i]} RMSE: {float_to_xlsx_string(errors[i])}\n")
            
            average_rmse: float = sum(errors) / len(errors)
            log_file.write(f"Average RMSE: {float_to_xlsx_string(average_rmse)} degrees\n\n")
        dir_num += 1
        dir_path  = upper_path + str(dir_num)
        print("\n")

print("Done")
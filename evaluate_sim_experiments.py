import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm

def overall_rmse(rmse_list, T=200):
    return np.sqrt((1 / (len(rmse_list) * T)) * np.sum(T * np.array(rmse_list)**2))

gtpath = "/workspace/raid/krb/boreas/robotica_sims"

sequences = sorted([f for f in os.listdir(gtpath)])
N = len(sequences) 

# print("EVALUATING DISCRETELIO...")
# resultpath = "/home/krb/ASRL/temp/steam_icp/boreas_velodyne/discretelio/"
T = 200
# slow = []
# medium = []
# fast = []
# slow_failures = 0
# medium_failures = 0
# fast_failures = 0
# for i, seq in tqdm(enumerate(sequences)):
#     try:
#         gt_seq_path = os.path.join(gtpath, seq, "applanix", "lidar_poses_tum.txt")
#         result_seq = f"{seq}_tum.txt"
#         result_seq_path = os.path.join(resultpath, result_seq)
#         arg = f"evo_ape tum {gt_seq_path} {result_seq_path} -a"
#         # print(arg)
#         with open("/tmp/evo_result.txt", "w") as f:
#             p = subprocess.Popen(arg.split(), stdout=f)
#             p.wait()
#             # print(p.returncode)
#         with open("/tmp/evo_result.txt") as f:
#             line = [line for line in f.readlines() if "rmse" in line]
#             rmse = float(line[0].split()[1])
#             # print(rmse)
#         if "slow" in seq:
#             slow.append(rmse)
#         if "medium" in seq:
#             medium.append(rmse)
#         if "fast" in seq:
#             fast.append(rmse)
#     except Exception as e:
#         print(e)
#         if "slow" in seq:
#             slow_failures += 1 
#         if "medium" in seq:
#             medium_failures += 1 
#         if "fast" in seq:
#             fast_failures += 1 

# print(f"SLOW   - mean    RMSE: {np.mean(slow)}")
# print(f"SLOW   - overall RMSE: {overall_rmse(slow, T)}")
# print(f"SLOW   - failure rate: {100.0 * slow_failures / 20.0} %")
# print(f"MEDIUM - mean    RMSE: {np.mean(medium)}")
# print(f"MEDIUM - overall RMSE: {overall_rmse(medium, T)}")
# print(f"MEDIUM - failure rate: {100.0 * medium_failures / 20.0} %")
# print(f"FAST   - mean    RMSE: {np.mean(fast)}")
# print(f"FAST   - overall RMSE: {overall_rmse(fast, T)}")
# print(f"FAST   - failure rate: {100.0 * fast_failures / 20.0} %")


# print("EVALUATING SINGERLIO...")
# resultpath = "/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlio/"

# slow = []
# medium = []
# fast = []
# slow_failures = 0
# medium_failures = 0
# fast_failures = 0
# for i, seq in tqdm(enumerate(sequences)):
#     try:
#         gt_seq_path = os.path.join(gtpath, seq, "applanix", "lidar_poses_tum.txt")
#         result_seq = f"{seq}_tum.txt"
#         result_seq_path = os.path.join(resultpath, result_seq)
#         arg = f"evo_ape tum {gt_seq_path} {result_seq_path} -a"
#         # print(arg)
#         with open("/tmp/evo_result.txt", "w") as f:
#             p = subprocess.Popen(arg.split(), stdout=f)
#             p.wait()
#             # print(p.returncode)
#         with open("/tmp/evo_result.txt") as f:
#             line = [line for line in f.readlines() if "rmse" in line]
#             rmse = float(line[0].split()[1])
#             # print(rmse)
#         if "slow" in seq:
#             slow.append(rmse)
#         if "medium" in seq:
#             medium.append(rmse)
#         if "fast" in seq:
#             fast.append(rmse)
#     except Exception as e:
#         print(e)
#         if "slow" in seq:
#             slow_failures += 1 
#         if "medium" in seq:
#             medium_failures += 1 
#         if "fast" in seq:
#             fast_failures += 1

# print(f"SLOW   - mean    RMSE: {np.mean(slow)}")
# print(f"SLOW   - overall RMSE: {overall_rmse(slow, T)}")
# print(f"SLOW   - failure rate: {100.0 * slow_failures / 20.0} %")
# print(f"MEDIUM - mean    RMSE: {np.mean(medium)}")
# print(f"MEDIUM - overall RMSE: {overall_rmse(medium, T)}")
# print(f"MEDIUM - failure rate: {100.0 * medium_failures / 20.0} %")
# print(f"FAST   - mean    RMSE: {np.mean(fast)}")
# print(f"FAST   - overall RMSE: {overall_rmse(fast, T)}")
# print(f"FAST   - failure rate: {100.0 * fast_failures / 20.0} %")


print("EVALUATING SINGERLOG...")
resultpath = "/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlog/"

slow = []
medium = []
fast = []
slow_failures = 0
medium_failures = 0
fast_failures = 0
for i, seq in tqdm(enumerate(sequences)):
    try:
        gt_seq_path = os.path.join(gtpath, seq, "applanix", "lidar_poses_tum.txt")
        result_seq = f"{seq}_tum.txt"
        result_seq_path = os.path.join(resultpath, result_seq)
        arg = f"evo_ape tum {gt_seq_path} {result_seq_path}"
        # print(arg)
        with open("/tmp/evo_result.txt", "w") as f:
            p = subprocess.Popen(arg.split(), stdout=f)
            p.wait()
            # print(p.returncode)
        with open("/tmp/evo_result.txt") as f:
            line = [line for line in f.readlines() if "rmse" in line]
            rmse = float(line[0].split()[1])
            # print(rmse)
        if "slow" in seq:
            slow.append(rmse)
        if "medium" in seq:
            medium.append(rmse)
        if "fast" in seq:
            fast.append(rmse)
    except Exception as e:
        print(e)
        if "slow" in seq:
            slow_failures += 1 
        if "medium" in seq:
            medium_failures += 1 
        if "fast" in seq:
            fast_failures += 1

print(f"SLOW   - mean    RMSE: {np.mean(slow)}")
print(f"SLOW   - overall RMSE: {overall_rmse(slow, T)}")
print(f"SLOW   - failure rate: {100.0 * slow_failures / 20.0} %")
print(f"MEDIUM - mean    RMSE: {np.mean(medium)}")
print(f"MEDIUM - overall RMSE: {overall_rmse(medium, T)}")
print(f"MEDIUM - failure rate: {100.0 * medium_failures / 20.0} %")
print(f"FAST   - mean    RMSE: {np.mean(fast)}")
print(f"FAST   - overall RMSE: {overall_rmse(fast, T)}")
print(f"FAST   - failure rate: {100.0 * fast_failures / 20.0} %")


print("EVALUATING SINGERLO...")
resultpath = "/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlo/"

slow = []
medium = []
fast = []
slow_failures = 0
medium_failures = 0
fast_failures = 0
for i, seq in tqdm(enumerate(sequences)):
    try:
        gt_seq_path = os.path.join(gtpath, seq, "applanix", "lidar_poses_tum.txt")
        result_seq = f"{seq}_tum.txt"
        result_seq_path = os.path.join(resultpath, result_seq)
        arg = f"evo_ape tum {gt_seq_path} {result_seq_path}"
        # print(arg)
        with open("/tmp/evo_result.txt", "w") as f:
            p = subprocess.Popen(arg.split(), stdout=f)
            p.wait()
            # print(p.returncode)
        with open("/tmp/evo_result.txt") as f:
            line = [line for line in f.readlines() if "rmse" in line]
            rmse = float(line[0].split()[1])
            # print(rmse)
        if "slow" in seq:
            slow.append(rmse)
        if "medium" in seq:
            medium.append(rmse)
        if "fast" in seq:
            fast.append(rmse)
    except Exception as e:
        print(e)
        if "slow" in seq:
            slow_failures += 1 
        if "medium" in seq:
            medium_failures += 1 
        if "fast" in seq:
            fast_failures += 1

print(f"SLOW   - mean    RMSE: {np.mean(slow)}")
print(f"SLOW   - overall RMSE: {overall_rmse(slow, T)}")
print(f"SLOW   - failure rate: {100.0 * slow_failures / 20.0} %")
print(f"MEDIUM - mean    RMSE: {np.mean(medium)}")
print(f"MEDIUM - overall RMSE: {overall_rmse(medium, T)}")
print(f"MEDIUM - failure rate: {100.0 * medium_failures / 20.0} %")
print(f"FAST   - mean    RMSE: {np.mean(fast)}")
print(f"FAST   - overall RMSE: {overall_rmse(fast, T)}")
print(f"FAST   - failure rate: {100.0 * fast_failures / 20.0} %")

import os
import numpy as np
import subprocess


sequences = sorted([f for f in os.listdir("/workspace/raid/krb/boreas/robotica_sims")])
N = len(sequences) 
for seq in sequences:
    print(seq)

print("RUNNING DISCRETELIO...")

# DiscreteLIO
for i, seq in enumerate(sequences):
    print(f"SEQUENCE: {seq} ({i + 1} / {N})")
    arg = "ros2 run steam_icp steam_icp --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/config/boreas_simulation_discretelio_config.yaml"
    arg += " -p dataset_options.root_path:=/workspace/raid/krb/boreas/robotica_sims"
    arg += f" -p dataset_options.sequence:={seq}"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=None)
    p.wait()
    print(p.returncode)

print("RUNNING SINGERLIO...")

# SingerLIO
for i, seq in enumerate(sequences):
    print(f"SEQUENCE: {seq} ({i + 1} / {N})")
    arg = "ros2 run steam_icp steam_icp --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/config/boreas_simulation_steamlio_config.yaml"
    arg += " -p output_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlio"
    arg += " -p log_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlio"
    arg += " -p odometry_options.debug_path:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlio"
    arg += " -p dataset_options.root_path:=/workspace/raid/krb/boreas/robotica_sims"
    arg += f" -p dataset_options.sequence:={seq}"
    arg += " -p odometry_options.steam.use_imu:=true"
    arg += " -p odometry_options.steam.use_accel:=true"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=None)
    p.wait()
    print(p.returncode)

print("RUNNING SINGERLOG...")

# SingerLOG
for i, seq in enumerate(sequences):
    print(f"SEQUENCE: {seq} ({i + 1} / {N})")
    arg = "ros2 run steam_icp steam_icp --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/config/boreas_simulation_steamlio_config.yaml"
    arg += " -p output_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlog"
    arg += " -p log_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlog"
    arg += " -p odometry_options.debug_path:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlog"
    arg += " -p dataset_options.root_path:=/workspace/raid/krb/boreas/robotica_sims"
    arg += f" -p dataset_options.sequence:={seq}"
    arg += " -p odometry_options.steam.use_imu:=true"
    arg += " -p odometry_options.steam.use_accel:=false"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=None)
    p.wait()
    print(p.returncode)

print("RUNNING SINGERLO...")

# SingerLO
for i, seq in enumerate(sequences):
    print(f"SEQUENCE: {seq} ({i + 1} / {N})")
    arg = "ros2 run steam_icp steam_icp --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/config/boreas_simulation_steamlio_config.yaml"
    arg += " -p output_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlo"
    arg += " -p log_dir:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlo"
    arg += " -p odometry_options.debug_path:=/home/krb/ASRL/temp/steam_icp/boreas_velodyne/singerlo"
    arg += " -p dataset_options.root_path:=/workspace/raid/krb/boreas/robotica_sims"
    arg += f" -p dataset_options.sequence:={seq}"
    arg += " -p odometry_options.steam.use_imu:=false"
    arg += " -p odometry_options.steam.use_accel:=false"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=None)
    p.wait()
    print(p.returncode)

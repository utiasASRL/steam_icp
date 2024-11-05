import os
import numpy as np
from copy import deepcopy
import subprocess


# ros2 run steam_icp steam_icp --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/config/ncd_steamlo_config.yaml -p odometry_options.steam.use_imu:=false -p odometry_options.steam.use_accel:=false -p odometry_options.steam.use_line_search:=false -p dataset_options.lidar_timestamp_round:=true -p dataset_options.lidar_timestamp_round_hz:=10000.0 -p log_dir:=/home/krb/ASRL/temp/steam_icp/experiments/37 -p output_dir:=/home/krb/ASRL/temp/steam_icp/experiments/37

base_arg = "ros2 run steam_icp simulation --ros-args --params-file /home/krb/Documents/steam_icp/steam_icp/simulation/sim.yaml"
N = 20

np.random.seed(42)


# Slow
for i in range(N):
    print(f"SLOW SIMULATION {i + 1} / {N}")
    arg = deepcopy(base_arg)
    arg += f" -p output_dir:=/workspace/raid/krb/boreas/robotica_sims/slow_{i:04}"
    # v0 = np.random.uniform(0.5, 1.0)
    # w0 = np.random.uniform(0.1, 0.5)
    v0 = 0.0
    w0 = 0.0
    arg += f" -p x0:=[0.,0.,0.,0.,0.,0.,{v0},0.,0.,0.,0.,{w0},0.0,0.,0.,0.,0.,0.0]"
    la = body_centric_linear_velocity_amplitudes = np.random.uniform(0.1, 0.5, size=3)
    aa = body_centric_angular_velocity_amplitudes = np.random.uniform(0.1, 0.5, size=3)
    arg += f" -p v_amps:=[{la[0]},{la[1]},{la[2]},{0.0},{0.0},{0.0}]"
    lf = body_centric_linear_velocity_frequencies = np.random.uniform(0.5, 1.0, size=3)
    af = body_centric_angular_velocity_frequencies = np.random.uniform(1.0, 2.0, size=3)
    arg += f" -p v_freqs:=[{lf[0]},{lf[1]},{lf[2]},{0.0},{0.0},{0.0}]"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=subprocess.PIPE)
    p.wait()
    print(p.returncode)

# Medium
for i in range(N):
    print(f"MEDIUM SIMULATION {i + 1} / {N}")
    arg = deepcopy(base_arg)
    arg += f" -p output_dir:=/workspace/raid/krb/boreas/robotica_sims/medium_{i:04}"
    # v0 = np.random.uniform(1.0, 2.0)
    # w0 = np.random.uniform(0.5, 1.0)
    v0 = 0.0
    w0 = 0.0
    arg += f" -p x0:=[0.,0.,0.,0.,0.,0.,{v0},0.,0.,0.,0.,{w0},0.0,0.,0.,0.,0.,0.0]"
    la = body_centric_linear_velocity_amplitudes = np.random.uniform(0.5, 1.0, size=3)
    aa = body_centric_angular_velocity_amplitudes = np.random.uniform(0.5, 1.0, size=3)
    arg += f" -p v_amps:=[{la[0]},{la[1]},{la[2]},{aa[0]},{aa[1]},{aa[2]}]"
    lf = body_centric_linear_velocity_frequencies = np.random.uniform(1.0, 2.0, size=3)
    af = body_centric_angular_velocity_frequencies = np.random.uniform(2.0, 4.0, size=3)
    arg += f" -p v_freqs:=[{lf[0]},{lf[1]},{lf[2]},{af[0]},{af[1]},{af[2]}]"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=subprocess.PIPE)
    p.wait()
    print(p.returncode)

# Fast
for i in range(N):
    print(f"FAST SIMULATION {i + 1} / {N}")
    arg = deepcopy(base_arg)
    arg += f" -p output_dir:=/workspace/raid/krb/boreas/robotica_sims/fast_{i:04}"
    # v0 = np.random.uniform(2.0, 3.0)
    # w0 = np.random.uniform(1.0, 2.0)
    v0 = 0.0
    w0 = 0.0
    arg += f" -p x0:=[0.,0.,0.,0.,0.,0.,{v0},0.,0.,0.,0.,{w0},0.0,0.,0.,0.,0.,0.0]"
    la = body_centric_linear_velocity_amplitudes = np.random.uniform(1.0, 2.0, size=3)
    aa = body_centric_angular_velocity_amplitudes = np.random.uniform(1.0, 2.0, size=3)
    arg += f" -p v_amps:=[{la[0]},{la[1]},{la[2]},{aa[0]},{aa[1]},{aa[2]}]"
    lf = body_centric_linear_velocity_frequencies = np.random.uniform(2.0, 4.0, size=3)
    af = body_centric_angular_velocity_frequencies = np.random.uniform(4.0, 8.0, size=3)
    arg += f" -p v_freqs:=[{lf[0]},{lf[1]},{lf[2]},{af[0]},{af[1]},{af[2]}]"
    print(arg)
    p = subprocess.Popen(arg.split(), stdout=subprocess.PIPE)
    p.wait()
    print(p.returncode)
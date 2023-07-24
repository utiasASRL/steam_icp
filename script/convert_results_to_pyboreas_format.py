import os
import os.path as osp
import argparse
from pathlib import Path
import numpy as np
import csv

def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = T.copy()
    T2[:3, :3] = T2[:3, :3].transpose()
    T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
    return T2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # example:
    # python3 convert_results_to_pyboreas_format.py --root /workspace/raid/krb/boreas/boreas-2021-09-02-11-42/ --inpath /home/krb/ASRL/temp/steam_icp/boreas_velodyne/steam/boreas-2021-09-02-11-42_poses.txt --outpath /home/krb/ASRL/temp/steam_icp/boreas_velodyne/steam/odometry_result/boreas-2021-09-02-11-42.txt
    parser.add_argument('--root', default='/raid/krb/boreas/boreas-2021-09-02-11-42',
                        type=str, help='location of root folder containing the directories of each sequence')
    parser.add_argument('--inpath', type=str, help='path to steam_icp result file')
    parser.add_argument('--outpath', type=str, help='path to output pyboreas-formatted result file')
    parser.add_argument('--sensor', default='lidar', type=str)
    args = parser.parse_args()
    root = args.root

    os.makedirs(Path(args.inpath).parent, exist_ok=True)

    lids = sorted([f for f in os.listdir(osp.join(root, 'lidar')) if f.endswith('.bin')])

    if args.sensor == 'lidar':
        T_a_s = np.loadtxt(osp.join(root, 'calib', 'T_applanix_lidar.txt'))
    elif args.sensor == 'radar':
        T_a_l = np.loadtxt(osp.join(root, 'calib', 'T_applanix_lidar.txt'))
        T_r_l = np.loadtxt(osp.join(root, 'calib', 'T_radar_lidar.txt'))
        T_a_s = T_a_l @ get_inverse_tf(T_r_l)

    yfwd2xfwd = np.array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64).reshape(4, 4)
    T_s_r = get_inverse_tf(yfwd2xfwd @ T_a_s)
    T_a_r = T_a_s @ T_s_r
    T_r_a = get_inverse_tf(T_a_r)

    lines = []
    with open(args.inpath, 'r') as f:
        lines = f.readlines()
    assert len(lids) == len(lines)

    result = []
    for lid, line in zip(lids, lines):
        tstamp = Path(lid).stem
        line_split = line.strip().split()
        # assert tstamp == line_split[0], "{} != {}".format(tstamp, line_split[0])
        values = [float(v) for v in line_split]
        T_m_s = np.eye(4, dtype=np.float64)
        T_m_s[0, 0:4] = values[0:4]
        T_m_s[1, 0:4] = values[4:8]
        T_m_s[2, 0:4] = values[8:12]
        T_ak_a0 = T_a_s @ get_inverse_tf(T_m_s) @ T_r_a
        T_a_w_res = T_ak_a0.flatten().tolist()[:12]
        result.append([int(tstamp)] + T_a_w_res)
        
    with open(args.outpath, 'w') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(result)
        print("Written to file: {}".format(args.outpath))


import numpy as np

from pyboreas import BoreasDataset
from pyboreas.utils.utils import load_lidar, enforce_orthog, get_inverse_tf, yawPitchRollToRot

import io
import PIL.Image
from pyboreas.utils.odometry import get_sequence_poses, get_sequence_poses_gt, get_path_from_Tvi_list

import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import os.path as osp


font = ImageFont.truetype("DejaVuSans.ttf", 72)
tmesh = o3d.io.read_triangle_mesh('toyota.obj', True)
mesh = tmesh.scale(scale=0.035, center=np.array([0, 0, 0]).reshape(3, 1))
mesh.compute_vertex_normals()
T = np.eye(4)
T[:3, :3] = mesh.get_rotation_matrix_from_xyz((np.pi / 2, np.pi / 4 + np.pi, 0))
T[0, 3] = 0
T[1, 3] = 0
T[2, 3] = -2
mesh.transform(T)


def get_lidar_projection(path, dim=7, vdim=4, vmin=-25, vmax=0, cmap=cm.gist_rainbow):
    points = load_lidar(path, dim=dim)
    intensity = points[:, vdim]
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    intensity = np.clip(intensity, vmin, vmax)
    colors = cmap((intensity - vmin) / (vmax - vmin))[:, 0:3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render = vis.get_render_option()
    render.load_from_json("render_option.json")

    vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        o3d.io.read_pinhole_camera_parameters('view_option.json'))

    o3dimg = vis.capture_screen_float_buffer(do_render=True)
    out = np.array(o3dimg) * 255
    out = Image.fromarray(out.copy().astype(np.uint8))
    vis.destroy_window()
    return out

def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)

def plot_path(T_odom, T_gt, xlim=None, ylim=None):
    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                'axes.linewidth': 1.5, 'font.family': 'serif', 'pdf.fonttype': 42})
    path_odom, path_gt = get_path_from_Tvi_list(T_odom, T_gt)
    plt.figure(figsize=(10, 10), tight_layout=True, dpi=300)
    plt.grid(color='k', which='both', linestyle='--', alpha=0.75, dashes=(8.5, 8.5))
    plt.axes().set_aspect('equal')

    plt.plot(path_odom[:, 0], path_odom[:, 1], "b", linewidth=2.5, label="Estimate")
    plt.plot(path_gt[:, 0], path_gt[:, 1], "r", linewidth=2.5, label="Groundtruth")
    plt.plot(
        path_gt[0, 0],
        path_gt[0, 1],
        "ks",
        markerfacecolor="none",
        label="Sequence Start",
    )
    plt.xlabel("x [m]", fontsize=16)
    plt.ylabel("y [m]", fontsize=16)
    plt.axis("equal")
    plt.legend(loc="upper left", edgecolor='k', fancybox=False)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    out = convert_plt_to_img()
    plt.close()
    return out

def read_traj_file(path):
    """Reads trajectory from a space-separated txt file
    Args:
        path (string): file path including file name
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as file:
        # read each time and pose to lists
        poses = []

        for line in file:
            line_split = line.strip().split()
            values = [float(v) for v in line_split]
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(enforce_orthog(pose))
    return poses

def convert_line_to_pose(line, dim=3):
    """Reads trajectory from list of strings (single row of the comma-separeted groundtruth file). See Boreas
    documentation for format
    Args:
        line (List[string]): list of strings
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (np.ndarray): 4x4 SE(3) pose
        (int): time in nanoseconds
    """
    # returns T_iv
    line = line.replace("\n", ",").split(",")
    # line = line.split()
    line = [float(i) for i in line[:-1]]
    # x, y, z -> 1, 2, 3
    # roll, pitch, yaw -> 7, 8, 9
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = line[1]  # x
    T[1, 3] = line[2]  # y
    # Note, yawPitchRollToRot returns C_v_i, where v is vehicle/sensor frame and i is stationary frame
    # For SE(3) state, we want C_i_v (to match r_i loaded above), and so we take transpose
    if dim == 3:
        T[2, 3] = line[3]  # z
        T[:3, :3] = yawPitchRollToRot(line[9], line[8], line[7]).T
    elif dim == 2:
        T[:3, :3] = yawPitchRollToRot(
            line[9],
            np.round(line[8] / np.pi) * np.pi,
            np.round(line[7] / np.pi) * np.pi,
        ).T
    else:
        raise ValueError(
            "Invalid dim value in convert_line_to_pose. Use either 2 or 3."
        )
    time = int(line[0])
    return T, time

def read_traj_file_gt(path, T_ab, dim):
    """Reads trajectory from a space-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses (from world to sensor frame)
        (List[int]): list of times in microseconds
    """
    with open(path, "r") as f:
        lines = f.readlines()
    poses = []
    times = []

    T_ab = enforce_orthog(T_ab)
    for line in lines[1:]:
        pose, time = convert_line_to_pose(line, dim)
        poses += [
            enforce_orthog(T_ab @ get_inverse_tf(pose))
        ]  # convert T_iv to T_vi and apply calibration
        times += [int(time)]  # microseconds
    return poses, times



if __name__ == '__main__':
    seqs = ["boreas-2022-08-05-12-59"]
    bd = BoreasDataset(
        root="/workspace/nas/ASRL/2021-Boreas/",
        split=[seqs],
    )
    seq = bd.sequences[0]
    seq.synchronize_frames(ref="aeva")
    save = '/home/krb/icra23video/'

    pred_path = "/workspace/Documents/steam_icp/results/aeva_long/steam_dicp/"
    seqs = ["boreas-2022-08-05-12-59.txt"]
    T_pred = read_traj_file(osp.join(pred_path, seqs[0]))
    dim = 3
    # T_gt, _, _, _ = get_sequence_poses_gt("/workspace/nas/ASRL/2021-Boreas/", seqs, dim)
    T_calib = np.loadtxt(os.path.join("/workspace/nas/ASRL/2021-Boreas/", "boreas-2022-08-05-12-59", "calib/T_applanix_aeva.txt"))
    T_gt, _ = read_traj_file_gt("/workspace/nas/ASRL/2021-Boreas/applanix/aeva_poses.csv", T_calib, dim)

    path_odom, path_gt = get_path_from_Tvi_list(T_pred, T_gt)
    xmin = min(np.min(path_odom[:, 0]), np.min(path_gt[:, 0])) - 250.0
    xmax = max(np.max(path_odom[:, 0]), np.max(path_gt[:, 0])) + 250.0
    ymin = min(np.min(path_odom[:, 1]), np.min(path_gt[:, 1])) - 250.0
    ymax = max(np.max(path_odom[:, 1]), np.max(path_gt[:, 1])) + 250.0

    i = 0
    for aev, cam in zip(seq.aeva_frames, seq.camera_frames):
        print(i)
        i += 1
        lidar = get_lidar_projection(aev.path)
        camera = Image.open(cam.path)
        path = plot_path(T_pred[:i], T_gt[:i], xlim=[xmin, xmax], ylim=[ymin, ymax])

        canvas = Image.new('RGB', (3840, 2160), color=(26, 26, 26))
        camera.thumbnail([1930, 1615])
        canvas.paste(camera, (0, 0))
        canvas.paste(path, (2345, 0))
        canvas.paste(lidar, (1930, 1080))
        draw = ImageDraw.Draw(canvas)
        canvas.save(osp.join(save, 'im{:06}.png'.format(i)))

        

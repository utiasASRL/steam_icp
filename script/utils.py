import numpy as np
import numpy.linalg as npla

from pylgmath import se3op, Transformation
from pysteam.problem import OptimizationProblem, StaticNoiseModel, L2LossFunc, WeightedLeastSquareCostTerm
from pysteam.solver import GaussNewtonSolver
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.p2p import P2PErrorEvaluator as p2p_error


def rotation_error(T, dim):
    if dim == 2:
        T_vec = se3op.tran2vec(T)
        T_vec[2:5] = 0.0  # z and roll, pitch to 0
        T = se3op.vec2tran(T_vec)
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(T, dim):
    if dim == 2:
        return npla.norm(T[:2, 3])
    return npla.norm(T[:3, 3])


def get_inverse_tf(T):
    T2 = T.copy()
    T2[:3, :3] = T2[:3, :3].transpose()
    T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
    return T2


def trajectory_distances(poses):
    dist = [0]
    for i in range(1, len(poses)):
        P1 = poses[i - 1]
        P2 = poses[i]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2 + dz**2))
    return dist


def last_frame_from_segment_length(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def get_avg_stats(errs):
    t_err = 0
    r_err = 0
    num = 0
    for err in errs:
        num += len(err)
        for e in err:
            t_err += e[2]
            r_err += e[1]
    t_err /= float(num)
    r_err /= float(num)

    return t_err * 100, r_err * 180 / np.pi


def get_stats(err, lengths):
    t_err = 0
    r_err = 0
    len2id = {x: i for i, x in enumerate(lengths)}
    t_err_len = [0.0]*len(len2id)
    r_err_len = [0.0]*len(len2id)
    len_count = [0]*len(len2id)
    for e in err:
        t_err += e[2]
        r_err += e[1]
        j = len2id[e[3]]
        t_err_len[j] += e[2]
        r_err_len[j] += e[1]
        len_count[j] += 1
    t_err /= float(len(err))
    r_err /= float(len(err))

    return t_err * 100, r_err * 180 / np.pi, \
        [a/float(b) * 100 for a, b in zip(t_err_len, len_count) if b != 0], \
        [a/float(b) * 180 / np.pi for a, b in zip(r_err_len, len_count) if b != 0]


def calc_sequence_errors(poses_gt, poses_pred, dim=3):
    step_size = 10
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    # Pre-compute distances from ground truth as reference
    dist = trajectory_distances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = get_inverse_tf(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_res = get_inverse_tf(poses_pred[first_frame]) @ poses_pred[last_frame]
            pose_error = get_inverse_tf(pose_delta_res) @ pose_delta_gt
            r_err = rotation_error(pose_error, dim)
            t_err = translation_error(pose_error, dim)
            err.append([first_frame, r_err/float(length), t_err/float(length), length])
    return err, lengths

def get_avg_rpe(pose_errors):
    t_err = np.sqrt(np.mean(np.array([translation_error(e, 3) for e in pose_errors]) ** 2))
    r_err = np.mean(np.array([rotation_error(e, 3) * 180.0 / np.pi for e in pose_errors]))
    return t_err, r_err


def evaluate_odometry_rpe(poses_gt, poses_pred):
    assert len(poses_gt) == len(poses_pred)

    pose_errors = []
    for i in range(1, len(poses_gt)):
        pose_delta_gt = get_inverse_tf(poses_gt[i - 1]) @ poses_gt[i]
        pose_delta_res = get_inverse_tf(poses_pred[i - 1]) @ poses_pred[i]
        pose_error = get_inverse_tf(pose_delta_res) @ pose_delta_gt
        pose_errors.append(pose_error)

    t_err_2d = np.sqrt(np.mean(np.array([translation_error(e, 2) for e in pose_errors]) ** 2))
    r_err_2d = np.mean(np.array([rotation_error(e, 2) * 180.0 / np.pi for e in pose_errors]))
    t_err = np.sqrt(np.mean(np.array([translation_error(e, 3) for e in pose_errors]) ** 2))
    r_err = np.mean(np.array([rotation_error(e, 3) * 180.0 / np.pi for e in pose_errors]))
    return t_err_2d, r_err_2d, t_err, r_err, pose_errors


def evaluate_odometry_kitti(gt_poses, pred_poses):
    assert len(gt_poses) == len(pred_poses)

    err_2d, path_lengths = calc_sequence_errors(gt_poses, pred_poses, 2)
    t_err_2d, r_err_2d, _, _ = get_stats(err_2d, path_lengths)
    err_3d, path_lengths = calc_sequence_errors(gt_poses, pred_poses, 3)
    t_err, r_err, _, _ = get_stats(err_3d, path_lengths)
    return t_err_2d, r_err_2d, t_err, r_err, err_3d


def align_path(T_mr_gt, T_mr_pred):
  T_gt_pred = SE3StateVar(Transformation(T_ba=np.eye(4)))

  noise_model = StaticNoiseModel(np.eye(3))
  loss_func = L2LossFunc()
  cost_terms = []
  for idx in range(len(T_mr_gt)):
    error_func = p2p_error(T_gt_pred, T_mr_gt[idx, :, 3:], T_mr_pred[idx, :, 3:])
    cost_terms.append(WeightedLeastSquareCostTerm(error_func, noise_model, loss_func))

  opt_prob = OptimizationProblem()
  opt_prob.add_state_var(T_gt_pred)
  opt_prob.add_cost_term(*cost_terms)

  gauss_newton = GaussNewtonSolver(opt_prob, verbose=False, max_iterations=100)
  gauss_newton.optimize()

  return T_gt_pred.value.matrix()


def add_plot_pose(ax, filename, label=None):
    error = np.loadtxt(filename)
    timestamps = error[:, 1] / 1e9
    T_rm = error[:, 2:18].reshape(-1, 4, 4)
    T_mr_vec = se3op.tran2vec(T_rm).squeeze()

    ax[0, 0].plot(timestamps, T_mr_vec[:, 0], label=label)
    ax[1, 0].plot(timestamps, T_mr_vec[:, 1], label=label)
    ax[2, 0].plot(timestamps, T_mr_vec[:, 2], label=label)
    ax[0, 1].plot(timestamps, T_mr_vec[:, 3+0], label=label)
    ax[1, 1].plot(timestamps, T_mr_vec[:, 3+1], label=label)
    ax[2, 1].plot(timestamps, T_mr_vec[:, 3+2], label=label)


def add_plot_velocity(ax, filename, label=None):
    error = np.loadtxt(filename)
    timestamps = error[:, 1] / 1e9
    w_mr_inr = error[:, 18:24]  # n by 6
    ax[0, 0].plot(timestamps, w_mr_inr[:, 0], label=label)
    ax[1, 0].plot(timestamps, w_mr_inr[:, 1], label=label)
    ax[2, 0].plot(timestamps, w_mr_inr[:, 2], label=label)
    ax[0, 1].plot(timestamps, w_mr_inr[:, 3+0], label=label)
    ax[1, 1].plot(timestamps, w_mr_inr[:, 3+1], label=label)
    ax[2, 1].plot(timestamps, w_mr_inr[:, 3+2], label=label)


def add_legend(ax, prefix = "velocity", xlabel='time [s]'):
    legend = [['x', 'y', 'z'], ['roll', 'pitch', 'yaw']]
    for i in range(3):
        for j in range(2):
            ax[i, j].set_xlabel(xlabel)
            ax[i, j].set_ylabel(prefix + ' ' + legend[j][i])
            ax[i, j].legend()


def plot_error(ax, filename, label=None):
    error = np.loadtxt(filename)
    error = error[:, 2:8]  # n by 6
    # error = np.abs(error)
    print("Average error:", np.mean(error, axis=0), "Average abs error:", np.mean(np.abs(error), axis=0))
    ax[0, 0].plot(error[:, 0], label=label)
    ax[1, 0].plot(error[:, 1], label=label)
    ax[2, 0].plot(error[:, 2], label=label)
    ax[0, 1].plot(error[:, 3+0], label=label)
    ax[1, 1].plot(error[:, 3+1], label=label)
    ax[2, 1].plot(error[:, 3+2], label=label)

def plot_rte_tran_error(ax, filename, label=None):
    error = np.loadtxt(filename)
    error = error[:, 2:8]  # n by 6
    # error = np.abs(error)
    print("Average error:", np.mean(error, axis=0), "Average abs error:", np.mean(np.abs(error), axis=0))
    ax[0, 0].plot(error[:, 0], label=label)
    ax[1, 0].plot(error[:, 1], label=label)
    ax[2, 0].plot(error[:, 2], label=label)
    ax[0, 1].plot(error[:, 3+0], label=label)
    ax[1, 1].plot(error[:, 3+1], label=label)
    ax[2, 1].plot(error[:, 3+2], label=label)
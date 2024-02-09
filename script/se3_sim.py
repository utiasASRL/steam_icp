import abc
from typing import Optional, Tuple

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse as sp_sparse
from sksparse.cholmod import cholesky

from estimate_qc import (
    get_q_singer,
    get_q_wnoj,
    get_qinv_singer,
    get_qinv_wnoj,
    get_tran_singer,
    get_tran_wnoj,
    hz_to_ns,
    ns_to_sec,
)

from pylgmath import se3op, so3op


class SimConfig(object):
    def __init__(self, config_parser) -> None:
        """Simulation configuration

        Notes:
            qc (float): power spectral density of the underlying GP
            ad (float): 1 / l where l is the length-scale parameter of the Singer prior
            acc_ms2 (float): nominal acceleration [m/s^2]
            imu_rate_hz (int): simulation IMU rate [hz]
            gps_rate_hz (int): simulation GPS rate [hz]
            sim_time_s (float): length of simulation [s]
            x_init (np.ndarray): initial state (x, vx) to start sim
            sigma_meas_acc (float): std dev of acceleration measurement noise
            sigma_meas_pos (float): std dev of position measurement noise
        """
        if config_parser.get("qc") is not None:
            self.qcd = config_parser.get("qc")
        else:
            self.qcd = np.ones(6)
        self.ad = config_parser.get("ad")
        if config_parser.get("acc_ms2") is not None:
            self.acc_ms2 = config_parser.get("acc_ms2")
        else:
            self.acc_ms2 = 1.0
        if config_parser.get("imu_rate_hz") is not None:
            self.imu_rate_hz = config_parser.get("imu_rate_hz")
        else:
            self.imu_rate_hz = 100
        if config_parser.get("gps_rate_hz") is not None:
            self.gps_rate_hz = config_parser.get("gps_rate_hz")
        else:
            self.gps_rate_hz = 10
        if config_parser.get("sim_time_s") is not None:
            self.sim_time_s = config_parser.get("sim_time_s")
        else:
            self.sim_time_s = 1.0
        self.x_init = config_parser.get("x_init")
        if self.x_init is not None:
            self.x_init = np.array(self.x_init).reshape(-1, 1)
            if self.x_init.shape[0] != 2:
                raise RuntimeError("incorrect dims: x_init")
        self.sigma_meas_acc = config_parser.get("sigma_meas_acc")
        if config_parser.get("sigma_meas_acc") is not None:
            self.sigma_meas_acc = config_parser.get("sigma_meas_acc")
        else:
            self.sigma_meas_acc = 0.01
        if config_parser.get("sigma_meas_pos") is not None:
            self.sigma_meas_pos = config_parser.get("sigma_meas_pos")
        else:
            self.sigma_meas_pos = 0.01
        if config_parser.get("sigma_meas_vel") is not None:
            self.sigma_meas_vel = config_parser.get("sigma_meas_vel")
        else:
            self.sigma_meas_vel = 0.01
        if config_parser.get("sigma_input_acc") is not None:
            self.sigma_input_acc = config_parser.get("sigma_input_acc")
        else:
            self.sigma_input_acc = 0.01
        if config_parser.get("enable_noise") is not None:
            self.enable_noise = config_parser.get("enable_noise")
        else:
            self.enable_noise = True
        if config_parser.get("use_sparse") is not None:
            self.use_sparse = config_parser.get("use_sparse")
        else:
            self.use_sparse = False
        if config_parser.get("use_meas_vel") is not None:
            self.use_meas_vel = config_parser.get("use_meas_vel")
        else:
            self.use_meas_vel = False
        if (self.imu_rate_hz / self.gps_rate_hz) % 1 != 0:
            raise RuntimeError("imu rate must be integer multiple of gps rate")


class Simulator(abc.ABC):
    """Base class for simulators"""

    @abc.abstractclassmethod
    def forward(self):
        """Run forward simulation"""


class WNOJSimulator(Simulator):
    def __init__(self, config: SimConfig) -> None:
        """Initialize WNOJ simulator"""
        self._config: SimConfig = config

    def _get_tran(self, dt: float) -> np.ndarray:
        return get_tran_wnoj(dt)

    def _get_q(self, dt: float) -> np.ndarray:
        return get_q_wnoj(dt, self._config.qc)

    def _get_q_inv(self, dt: float) -> np.ndarray:
        return get_qinv_wnoj(dt, self._config.qc)

    def forward(
        self,
        seed=42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run forwards simulation, injecting noise using Qk"""
        rng = np.random.default_rng(seed)
        dt_imu_ns = hz_to_ns(self._config.imu_rate_hz)
        dt_s = ns_to_sec(dt_imu_ns)
        num_steps = int(self._config.sim_time_s * self._config.imu_rate_hz) + 1
        dim = 18
        states = np.zeros((num_steps, dim))  # TODO
        Phi = self._get_tran(dt_s)
        Qk_inv = self._get_q_inv(dt_s)
        states[0, :] = np.array([0, 0, self._config.acc_ms2])
        states_tns = np.zeros(num_steps, np.int64)
        if self._config.x_init is not None:
            states[0, :] = self._config.x_init[:]

        # TODO: produce x_check using global prior...

        # Ainv = (
        #     sp_sparse.lil_array(sp_sparse.eye(num_steps * dim))
        #     if self._config.use_sparse
        #     else np.eye(num_steps * dim)
        # )
        # for i in range(num_steps - 1):
        #     Ainv[(i + 1) * dim : (i + 2) * dim, i * dim : (i + 1) * dim] = -Phi
        
        x_check = np.zeros((num_steps, dim))
        x_check[0, :] = states[0, :dim]
        global_vars = []
        T_k0 = se3op.vec2tran(self._config.x_init[:6].reshape(-1, 1))
        w_0k_in_k = self._config.x_init[6:12].reshape(-1, 1)
        dw_0k_in_k = self._config.x_init[12:].reshape(-1, 1)
        global_vars.append([np.copy(T_k0), np.copy(w_0k_in_k), np.copy(dw_0k_in_k)])
        for i in range(1, num_steps):
            gamma1 = np.zeros((dim, 1))
            gamma1[6:12, 0:1] = w_0k_in_k
            gamma1[12:, 0:1] = dw_0k_in_k
            gamma2 = Phi @ gamma1
            xi_i1 = gamma2[:6].reshape(-1, 1)
            xi_j1 = gamma2[6:12].reshape(-1, 1)
            xi_k1 = gamma2[12:].reshape(-1, 1)
            T_k0 = se3op.vec2tran(xi_i1) @ T_k0
            J_21 = se3op.vec2jac(xi_i1)
            w_0k_in_k = J_21 @ gamma2[6:12].reshape(-1, 1)
            dw_0k_in_k = J_21 @ (xi_k1 + 0.5 * se3op.curlyhat(xi_j1) @ w_0k_in_k)
            global_vars.append([np.copy(T_k0), np.copy(w_0k_in_k), np.copy(dw_0k_in_k)])
        Qinv = (
            sp_sparse.lil_array(sp_sparse.block_diag(([Qk_inv] * num_steps)))
            if self._config.use_sparse
            else spla.block_diag(*([Qk_inv] * num_steps))
        )
        Qinv[:dim, :dim] = np.diag([1e3, 1e3, 1e3])
        P_check_inv = Ainv.T @ Qinv @ Ainv
        # Don't allow samples outside of 4-sigma (sample from truncated Gaussian)
        noise = rng.normal(size=num_steps * dim).reshape(-1, 1)
        for j in range(noise.shape[0]):
            while np.abs(noise[j, 0]) > 4.0:
                noise[j, 0] = rng.normal()
        if sp_sparse.issparse(P_check_inv):
            P_check_inv = 0.5 * (P_check_inv + P_check_inv.T)
            factor = cholesky(
                P_check_inv, beta=0, mode="simplicial", ordering_method="natural"
            )
            # P_check = factor.inv()
            # P_check = 0.5 * (P_check + P_check.T)
            # P_check = sp_sparse.csc_matrix(sp_sparse.linalg.inv(P_check_inv))
            # L = factor.L()
            # V = sp_sparse.linalg.inv(L).T
            y_meas = factor.solve_Lt(noise, use_LDLt_decomposition=False)

            # Cholesky factorization
            # factor2 = cholesky(
            #     P_check, beta=1.0e-10, mode="simplicial", ordering_method="amd"
            # )
            # V = factor2.L()
            # manually sample from multi-dimensional Gaussian

            # states = x_check.reshape(-1, 1) + V @ noise
            states = x_check.reshape(-1, 1) + y_meas
        else:
            P_check = npla.inv(P_check_inv)
            P_check = 0.5 * (P_check + P_check.T)
            V = npla.cholesky(P_check)
            states = x_check.reshape(-1, 1) + V @ noise
            # states = rng.multivariate_normal(x_check.reshape(-1), P_check)

        states = states.reshape(-1, dim)
        tns = dt_imu_ns
        for i in range(1, num_steps):
            states_tns[i] = tns
            tns += dt_imu_ns
        meas_acc = np.copy(states[:-1, 2])
        if self._config.enable_noise:
            mnoise = rng.normal(size=meas_acc.shape[0])
            for j in range(mnoise.shape[0]):
                while np.abs(mnoise[j]) > 4.0:
                    mnoise[j] = rng.normal()
            meas_acc += self._config.sigma_meas_acc * mnoise
        meas_acc_tns = states_tns[:-1]
        skip = int(self._config.imu_rate_hz / self._config.gps_rate_hz)
        meas_pos = np.copy(states[::skip, 0])
        if self._config.enable_noise:
            mnoise = rng.normal(size=meas_pos.shape[0])
            for j in range(mnoise.shape[0]):
                while np.abs(mnoise[j]) > 4.0:
                    mnoise[j] = rng.normal()
            meas_pos += self._config.sigma_meas_pos * mnoise
        meas_pos_tns = states_tns[::skip]
        meas_vel = np.array([])
        meas_vel_tns = np.array([])
        if self._config.use_meas_vel:
            meas_vel = np.copy(states[::skip, 1])
            if self._config.enable_noise:
                mnoise = rng.normal(size=meas_vel.shape[0])
                for j in range(mnoise.shape[0]):
                    while np.abs(mnoise[j]) > 4.0:
                        mnoise[j] = rng.normal()
                meas_vel += self._config.sigma_meas_vel * mnoise
            meas_vel_tns = states_tns[::skip]

        return (
            states,
            states_tns,
            meas_acc,
            meas_acc_tns,
            meas_pos,
            meas_pos_tns,
            P_check_inv,
            x_check,
            meas_vel,
            meas_vel_tns,
        )


class SingerSimulator(WNOJSimulator):
    def _get_tran(self, dt: float) -> np.ndarray:
        return get_tran_singer(dt, self._config.ad)

    def _get_q(self, dt: float) -> np.ndarray:
        return get_q_singer(dt, self._config.ad, self._config.qc)

    def _get_q_inv(self, dt: float) -> np.ndarray:
        return get_qinv_singer(dt, self._config.ad, self._config.qc)


class SimulatorFactory:
    def get_simulator(self, sim_type: str, config: SimConfig):
        if sim_type == "WNOJ":
            return WNOJSimulator(config)
        elif sim_type == "SINGER":
            return SingerSimulator(config)
        else:
            raise NotImplementedError(
                "simulation type {} not supported".format(sim_type)
            )
import numpy as np
import numpy.linalg as npla
import os
import os.path as osp
import argparse

# GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,roll,pitch,heading,angvel_z,angvel_y,angvel_x,accelz,accely,accelx,latitude,longitude

SBET_RATE = 200.0

def roll(r):
  return np.array(
      [[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]],
      dtype=np.float64,
  )


def pitch(p):
  return np.array(
      [[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]],
      dtype=np.float64,
  )


def yaw(y):
  return np.array(
      [[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]],
      dtype=np.float64,
  )


def yawPitchRollToRot(y, p, r):
  return roll(r) @ pitch(p) @ yaw(y)

def get_transform(gt):
  """Retrieve 4x4 homogeneous transform for a given parsed line of the ground truth pose csv
  Args:
      gt (List[float]): parsed line from ground truth csv file
  Returns:
      np.ndarray: 4x4 transformation matrix (pose of sensor)
  """
  T = np.identity(4, dtype=np.float64)
  C_enu_sensor = yawPitchRollToRot(gt[9], gt[8], gt[7])
  T[0, 3] = gt[1]
  T[1, 3] = gt[2]
  T[2, 3] = gt[3]
  T[0:3, 0:3] = C_enu_sensor
  return T

def get_qinv_wnoj(dt: float, qcd: np.ndarray = None):
  dtinv = 1.0 / dt

  qinv00 = 720.0 * (dtinv**5)
  qinv11 = 192.0 * (dtinv**3)
  qinv22 = 9.0 * (dtinv)
  qinv01 = qinv10 = (-360.0) * (dtinv**4)
  qinv02 = qinv20 = (60.0) * (dtinv**3)
  qinv12 = qinv21 = (-36.0) * (dtinv**2)

  if qcd is None:
    return np.array(
        [
            [qinv00, qinv01, qinv02],
            [qinv10, qinv11, qinv12],
            [qinv20, qinv21, qinv22],
        ]
    )
  Qcinv = np.diag(1.0 / qcd)
  return np.block(
      [
          [qinv00 * Qcinv, qinv01 * Qcinv, qinv02 * Qcinv],
          [qinv10 * Qcinv, qinv11 * Qcinv, qinv12 * Qcinv],
          [qinv20 * Qcinv, qinv21 * Qcinv, qinv22 * Qcinv],
      ]
  )

def get_q_wnoj(dt: float, qcd: np.ndarray = None):
  q00 = (dt**5) / 20.0
  q11 = (dt**3) / 3.0
  q22 = dt
  q01 = q10 = (dt**4) / 8.0
  q02 = q20 = (dt**3) / 6.0
  q12 = q21 = (dt**2) / 2.0

  if qcd is None:
    return np.array(
        [
            [q00, q01, q02],
            [q10, q11, q12],
            [q20, q21, q22],
        ]
    )

  Qc = np.diag(qcd)
  return np.block(
      [
          [q00 * Qc, q01 * Qc, q02 * Qc],
          [q10 * Qc, q11 * Qc, q12 * Qc],
          [q20 * Qc, q21 * Qc, q22 * Qc],
      ]
  )

def get_tran_wnoj(dt: float, dim=1) -> np.ndarray:

  assert dim >= 1, "invalid argument dim: {}".format(dim)
  I = np.eye(dim)
  O = np.zeros((dim, dim))
  return np.block(
      [
          [I, dt * I, 0.5 * dt**2 * I],
          [O, I, dt * I],
          [O, O, I],
      ]
  )
      
def get_q_singer(dt: float, add: np.ndarray, qcd: np.ndarray) -> np.ndarray:
  dim = add.squeeze().shape[0]
  assert dim == qcd.squeeze().shape[0]

  Q11 = np.zeros((dim, dim))
  Q12 = np.zeros((dim, dim))
  Q13 = np.zeros((dim, dim))
  Q22 = np.zeros((dim, dim))
  Q23 = np.zeros((dim, dim))
  Q33 = np.zeros((dim, dim))

  for i in range(dim):
    ad = add.squeeze()[i]
    qc = qcd.squeeze()[i]

    if np.abs(ad) >= 1.0:
      adi = 1.0 / ad
      adi2 = adi * adi
      adi3 = adi * adi2
      adi4 = adi * adi3
      adi5 = adi * adi4
      adt = ad * dt
      adt2 = adt * adt
      adt3 = adt2 * adt
      expon = np.exp(-adt)
      expon2 = np.exp(-2 * adt)
      Q11[i, i] = qc * (
          0.5
          * adi5
          * (1 - expon2 + 2 * adt + (2.0 / 3.0) * adt3 - 2 * adt2 - 4 * adt * expon)
      )
      Q12[i, i] = qc * (
          0.5 * adi4 * (expon2 + 1 - 2 * expon + 2 * adt * expon - 2 * adt + adt2)
      )
      Q13[i, i] = qc * 0.5 * adi3 * (1 - expon2 - 2 * adt * expon)
      Q22[i, i] = qc * 0.5 * adi3 * (4 * expon - 3 - expon2 + 2 * adt)
      Q23[i, i] = qc * 0.5 * adi2 * (expon2 + 1 - 2 * expon)
      Q33[i, i] = qc * 0.5 * adi * (1 - expon2)
    else:
      dt2 = dt * dt
      dt3 = dt * dt2
      dt4 = dt * dt3
      dt5 = dt * dt4
      dt6 = dt * dt5
      dt7 = dt * dt6
      dt8 = dt * dt7
      dt9 = dt * dt8
      ad2 = ad * ad
      ad3 = ad * ad2
      ad4 = ad * ad3
      # use Taylor series expansion about ad = 0
      Q11[i, i] = qc * (
          0.05 * dt5
          - 0.0277778 * dt6 * ad
          + 0.00992063 * dt7 * ad2
          - 0.00277778 * dt8 * ad3
          + 0.00065586 * dt9 * ad4
      )
      Q12[i, i] = qc * (
          0.125 * dt**4
          - 0.0833333 * dt5 * ad
          + 0.0347222 * dt6 * ad2
          - 0.0111111 * dt7 * ad3
          + 0.00295139 * dt8 * ad4
      )
      Q13[i, i] = qc * (
          (1 / 6) * dt3
          - (1 / 6) * dt4 * ad
          + 0.0916667 * dt5 * ad2
          - 0.0361111 * dt6 * ad3
          + 0.0113095 * dt7 * ad4
      )
      Q22[i, i] = qc * (
          (1 / 3) * dt3
          - 0.25 * dt4 * ad
          + 0.116667 * dt5 * ad2
          - 0.0416667 * dt6 * ad3
          + 0.0123016 * dt7 * ad4
      )
      Q23[i, i] = qc * (
          0.5 * dt2
          - 0.5 * dt3 * ad
          + 0.291667 * dt4 * ad2
          - 0.125 * dt5 * ad3
          + 0.0430556 * dt6 * ad4
      )
      Q33[i, i] = qc * (
          dt
          - dt2 * ad
          + (2 / 3) * dt3 * ad2
          - (1 / 3) * dt4 * ad3
          + 0.133333 * dt5 * ad4
      )

  return np.block(
      [
          [Q11, Q12, Q13],
          [Q12, Q22, Q23],
          [Q13, Q23, Q33],
      ]
  )

def get_jac_Qk_alpha(dt: float, add: np.ndarray, qcd: np.ndarray) -> np.ndarray:
  dim = add.squeeze().shape[0]
  assert dim == qcd.squeeze().shape[0]

  dt2 = dt * dt
  dt3 = dt * dt2
  dt4 = dt * dt3
  dt5 = dt * dt4

  dQda = np.zeros((dim, 3 * dim, 3 * dim))

  for i in range(dim):
    dQ11 = np.zeros((dim, dim))
    dQ12 = np.zeros((dim, dim))
    dQ13 = np.zeros((dim, dim))
    dQ22 = np.zeros((dim, dim))
    dQ23 = np.zeros((dim, dim))
    dQ33 = np.zeros((dim, dim))

    ad = add.squeeze()[i]
    qc = qcd.squeeze()[i]

    if np.abs(ad) <= 0.05:
      # print(
      #     "Partial Qk / Partial alpha goes to infty as alpha goes to zero, setting it to zero..."
      # )
      continue
    adi = 1 / ad
    adi2 = adi * adi
    adi3 = adi * adi2
    if np.abs(ad) <= 4.0:
      # Laurent/Taylor series approx
      dt6 = dt * dt5
      dt7 = dt * dt6
      dt8 = dt * dt7
      dt9 = dt * dt8
      dt10 = dt * dt9
      ad2 = ad * ad
      ad3 = ad * ad2
      ad4 = ad * ad3
      dQ11[i, i] = qc * (
          8.88178e-16 * dt3 * adi3
          - 1.33227e-15 * dt4 * adi2
          + 1.33227e-15 * dt5 * adi
          - 0.0277778 * dt6
          + 0.0198413 * dt7 * ad
          - 0.00844444 * dt8 * ad2
          + 0.00262346 * dt9 * ad3
          - 0.00067791 * dt10 * ad4
      )
      dQ12[i, i] = qc * (
          1.11022e-16 * dt4 * adi
          - 0.0833333 * dt5
          + 0.0694444 * dt6 * ad
          - 0.0333333 * dt7 * ad2
          + 0.0118056 * dt8 * ad3
          - 0.00338955 * dt9 * ad4
      )
      dQ13[i, i] = qc * (
          -0.166667 * dt4
          + 0.183333 * dt5 * ad
          - 0.108333 * dt6 * ad2
          + 0.045238 * dt7 * ad3
          - 0.01488 * dt8 * ad4
      )
      dQ22[i, i] = qc * (
          -0.25 * dt4
          + 0.233333 * dt5 * ad
          - 0.125 * dt6 * ad2
          + 0.0492063 * dt7 * ad3
          - 0.015625 * dt8 * ad4
      )
      dQ23[i, i] = qc * (
          -0.5 * dt3
          + 0.583333 * dt4 * ad
          - 0.375 * dt5 * ad2
          + 0.172222 * dt6 * ad3
          - 0.0625 * dt7 * ad4
      )
      dQ33[i, i] = qc * (
          -dt2
          + 1.33333 * dt3 * ad
          - dt4 * ad2
          + 0.533333 * dt5 * ad3
          - 0.222222 * dt6 * ad4
      )
    else:
      adi4 = adi * adi3
      adi5 = adi * adi4
      adi6 = adi * adi5
      expon = np.exp(-dt * ad)
      expon2 = np.exp(-2 * dt * ad)
      dQ11[i, i] = qc * (
          -(2 / 3) * dt3 * adi3
          + 2 * dt2 * expon * adi4
          + 3 * dt2 * adi4
          + 2.5 * expon2 * adi6
          + dt * expon2 * adi5
          + 8 * dt * expon * adi5
          - 4 * dt * adi5
          - 2.5 * adi6
      )
      dQ12[i, i] = qc * (
          -dt2 * expon * adi3
          - dt2 * adi3
          - 2 * expon2 * adi5
          + 4 * expon * adi5
          - dt * expon2 * adi4
          - 2 * dt * expon * adi4
          + 3 * dt * adi4
          - 2 * adi5
      )
      dQ13[i, i] = qc * (
          dt2 * expon * adi2
          + 1.5 * expon2 * adi4
          + dt * expon2 * adi3
          + 2 * dt * expon * adi3
          - 1.5 * adi4
      )
      dQ22[i, i] = qc * (
          1.5 * expon2 * adi4
          - 6 * expon * adi4
          + dt * expon2 * adi3
          - 2 * dt * expon * adi3
          - 2 * dt * adi3
          + 4.5 * adi4
      )
      dQ23[i, i] = qc * (
          -expon2 * adi3
          + 2 * expon * adi3
          - dt * expon2 * adi2
          + dt * expon * adi2
          - adi3
      )

      dQ33[i, i] = qc * (0.5 * expon2 * adi2 + dt * expon2 * adi - 0.5 * adi2)

    dQda[i] = np.block(
      [
          [dQ11, dQ12, dQ13],
          [dQ12, dQ22, dQ23],
          [dQ13, dQ23, dQ33],
      ]
    )
  return dQda

def get_jac_Q_sigma(qcd: np.ndarray) -> np.ndarray:
  dim = qcd.squeeze().shape[0]
  dQdsigma = np.zeros((dim, 3 * dim, 3 * dim))
  for i in range(dim):
    dQdsigma[i, i, i] = 1
    dQdsigma[i, dim + i, dim + i] = 1
    dQdsigma[i, 2 * dim + i, 2 * dim + i] = 1
  return dQdsigma

def get_jac_Qk_sigma(qcd: np.ndarray) -> np.ndarray:
  dim = qcd.squeeze().shape[0]
  dQdsigma = np.zeros((dim, 3 * dim, 3 * dim))
  for i in range(dim):
    dQdsigma[i, i, i] = 1
    dQdsigma[i, dim + i, dim + i] = 1
    dQdsigma[i, 2 * dim + i, 2 * dim + i] = 1
  return dQdsigma

def get_qinv_singer(dt: float, add: np.ndarray, qcd: np.ndarray) -> np.ndarray:
  return npla.inv(get_q_singer(dt, add, qcd))


def get_tran_singer(dt: float, add: np.ndarray) -> np.ndarray:
  dim = add.squeeze().shape[0]

  C1 = np.zeros((dim, dim))
  C2 = np.zeros((dim, dim))
  C3 = np.zeros((dim, dim))

  for i in range(dim):
    ad = add.squeeze()[i]
    if np.abs(ad) >= 1.0:
      adinv = 1.0 / ad
      adt = ad * dt
      expon: float = np.exp(-adt)
      C1[i, i] = (adt - 1.0 + expon) * adinv * adinv
      C2[i, i] = (1.0 - expon) * adinv
      C3[i, i] = expon
    else:
      C1[i, i] = (
          0.5 * dt**2
          - (1 / 6) * dt**3 * ad
          + (1 / 24) * dt**4 * ad**2
          - (1 / 120) * dt**5 * ad**3
          + (1 / 720) * dt**6 * ad**4
      )
      C2[i, i] = (
          dt
          - 0.5 * dt**2 * ad
          + (1 / 6) * dt**3 * ad**2
          - (1 / 24) * dt**4 * ad**3
          + (1 / 120) * dt**5 * ad**4
      )
      C3[i, i] = (
          1
          - dt * ad
          + 0.5 * dt**2 * ad**2
          - (1 / 6) * dt**3 * ad**3
          + (1 / 24) * dt**4 * ad**4
      )
  I = np.eye(dim)
  O = np.zeros((dim ,dim))
  return np.block(
      [
          [I, dt * I, C1],
          [O, I, C2],
          [O, O, C3],
      ]
  )

def get_jac_ek_alpha(dt: float, add: np.ndarray, xk: np.ndarray) -> np.ndarray:
  """e_k = x_{k+1} - A_k @ x_k

  Args:
      dt (float): time delta
      ad (float): alpha (1 / length-scale)
      xk (np.ndarray): 'previous' state

  Raises:
      RuntimeError: raised if ad <= 0.05 due to Jacobian approaching infty

  Returns:
      np.ndarray: Jacobian partial ek / partial alpha
  """
  dim = add.squeeze().shape[0]
  assert xk.shape[0] == dim * 3, "dimension of xk does not match add"
  de_da = np.zeros((dim, 3 * dim, 1))
  for i in range(dim):
    dC1 = np.zeros((dim, dim))
    dC2 = np.zeros((dim, dim))
    dC3 = np.zeros((dim, dim))
    ad = add.squeeze()[i]

    if np.abs(ad) >= 0.05:
      # print(
      #     "Partial ek / Partial alpha goes to infty as alpha goes to zero"
      # )
      expon = np.exp(-dt * ad)
      dC3[i, i] = -dt * expon
      if np.abs(ad) >= 1.0:
        adi = 1 / ad
        adi2 = adi * adi
        adi3 = adi * adi2
        dC1[i, i] = -2 * expon * adi3 - dt * expon * adi2 - dt * adi2 + 2 * adi3
        dC2[i, i] = expon * adi2 + dt * expon * adi - adi2
      else:
        ad2 = ad * ad
        ad3 = ad * ad2
        ad4 = ad * ad3
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3
        dt5 = dt * dt4
        dt6 = dt * dt5
        dt7 = dt * dt6
        dC1[i, i] = (
            -dt3 / 6
            + dt4 * ad / 12
            - dt5 * ad2 / 40
            - dt6 * ad3 / 180
            - dt7 * ad4 / 1008
        )
        dC2[i, i] = (
            -dt2 / 2 + dt3 * ad / 3 - dt4 * ad2 / 8 + dt5 * ad3 / 30 - dt6 * ad4 / 144
        )
      dC = np.block(
        [
          [dC1],
          [dC2],
          [dC3],
        ]
      )
      de_da[i] = -1 * dC @ xk.reshape(dim * 3, 1)[-dim:]
  return de_da

def get_jac_C_alpha(dt: float, add: np.ndarray) -> np.ndarray:
  """

  Args:
      dt (float): time delta
      ad (float): alpha (1 / length-scale)

  Raises:
      RuntimeError: raised if ad <= 0.05 due to Jacobian approaching infty

  Returns:
      np.ndarray: Jacobian partial ek / partial alpha
  """
  dim = add.squeeze().shape[0]
  dC_da = np.zeros((dim, 3 * dim, dim))

  for i in range(dim):
    dC1 = np.zeros((dim, dim))
    dC2 = np.zeros((dim, dim))
    dC3 = np.zeros((dim, dim))
    ad = add.squeeze()[i]
    if np.abs(ad) <= 0.05:
      print(
          "Partial ek / Partial alpha goes to infty as alpha goes to zero"
      )
      continue
    expon = np.exp(-dt * ad)
    dC3[i, i] = -dt * expon
    if np.abs(ad) >= 1.0:
      adi = 1 / ad
      adi2 = adi * adi
      adi3 = adi * adi2
      dC1[i, i] = -2 * expon * adi3 - dt * expon * adi2 - dt * adi2 + 2 * adi3
      dC2[i, i] = expon * adi2 + dt * expon * adi - adi2
    else:
      ad2 = ad * ad
      ad3 = ad * ad2
      ad4 = ad * ad3
      dt2 = dt * dt
      dt3 = dt * dt2
      dt4 = dt * dt3
      dt5 = dt * dt4
      dt6 = dt * dt5
      dt7 = dt * dt6
      dC1[i, i] = (
          -dt3 / 6
          + dt4 * ad / 12
          - dt5 * ad2 / 40
          - dt6 * ad3 / 180
          - dt7 * ad4 / 1008
      )
      dC2[i, i] = (
          -dt2 / 2 + dt3 * ad / 3 - dt4 * ad2 / 8 + dt5 * ad3 / 30 - dt6 * ad4 / 144
      )
    dC_da[i] = np.block(
      [
        [dC1],
        [dC2],
        [dC3],
      ]
    )
  return dC_da


def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = np.identity(4, dtype=T.dtype)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2

EARTH_SEMIMAJOR = 6378137.0
EARTH_SEMIMINOR = 6356752.0
EARTH_ECCEN     = 0.081819190842622
EARTH_ECCEN_2 = EARTH_ECCEN*EARTH_ECCEN
a = EARTH_SEMIMAJOR
eccSquared = EARTH_ECCEN**2
eccPrimeSquared = (eccSquared) / (1 - eccSquared)
k0 = 0.9996     # scale factor
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 1.0 / DEG_TO_RAD

def lla2ecef(lla):
    lat = lla[0]
    lon = lla[1]
    alt = lla[2]

    slat = np.sin(lat)
    s2lat = slat*slat
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)
    rprime = EARTH_SEMIMAJOR/np.sqrt(1.0 - EARTH_ECCEN_2*s2lat)

    # position
    x = (rprime + alt)*clat*clon
    y = (rprime + alt)*clat*slon
    z = ((1.0 - EARTH_ECCEN_2)*rprime + alt)*slat

    # rotation
    R_ned_ecef = np.eye(3, dtype=np.float64)
    R_ned_ecef[0, 0] = -slat * clon
    R_ned_ecef[0, 1] = -slat * slon
    R_ned_ecef[0, 2] = clat
    R_ned_ecef[1, 0] = -slon
    R_ned_ecef[1, 1] = clon
    R_ned_ecef[1, 2] = 0
    R_ned_ecef[2, 0] = -clat * clon
    R_ned_ecef[2, 1] = -clat * slon
    R_ned_ecef[2, 2] = -slat

    # output transform
    T_ecef_ned = np.eye(4, dtype=np.float64)
    T_ecef_ned[0, 3] = x
    T_ecef_ned[1, 3] = y
    T_ecef_ned[2, 3] = z
    T_ecef_ned[:3, :3] = R_ned_ecef.T

    return T_ecef_ned

def RelLLAtoNED(lla, ll0):
  # make sure ll0 shape is 3
  assert(ll0.ndim == 1)
  assert(ll0.shape[0] == 3)

  # make sure lla shape is 3
  assert(lla.ndim == 1)
  assert(lla.shape[0] == 3)

  # positions in ECEF frame
  T_ecef_ned = lla2ecef(lla)
  T_ecef_nedref = lla2ecef(ll0)

  # output
  T_nedref_ned = np.matmul(get_inverse_tf(T_ecef_nedref), T_ecef_ned)
  return T_nedref_ned

# X-Y-Z (roll then pitch then yaw) with EXTRINSIC ROTATIONS
# NOTE: they define their principal rotation matrices as the INVERSE of the ones we use at UTIAS
# so the final output needs to be TRANSPOSED to get what we would expect.
# Instead of this, you can simply use our own function: yawPitchRollToRot(y, p, r)
def posOrientToRot(heading, pitch, roll):
  theta = pitch
  phi = roll
  psi = heading

  ctheta = np.cos(theta)
  stheta = np.sin(theta)
  cphi   = np.cos(phi)
  sphi   = np.sin(phi)
  cpsi   = np.cos(psi)
  spsi   = np.sin(psi)

  R = np.identity(3, dtype=np.float64)

  R[0, 0] = ctheta * cpsi
  R[0, 1] = -cphi * spsi + sphi * stheta * cpsi
  R[0, 2] = sphi * spsi + cphi * stheta * cpsi
  R[1, 0] = ctheta * spsi
  R[1, 1] = cphi * cpsi + sphi * stheta * spsi
  R[1, 2] = -sphi * cpsi + cphi * stheta * spsi
  R[2, 0] = -stheta
  R[2, 1] = sphi * ctheta
  R[2, 2] = cphi * ctheta
  return R

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, help='path to boreas-2021-...', default='/workspace/raid/krb/boreas/boreas-2021-09-02-11-42')
  args = parser.parse_args()

  with open(osp.join(args.root, 'applanix', 'gps_post_process.csv')) as f:
    f.readline()
    lines = f.readlines()
  data = []
  for line in lines:
    data.append([float(x) for x in line.rstrip().split(',')])
  data = np.array(data)
  data = data[:, [0, 3, 2, 1, 6, 5, 4]]  # z,y,x --> x,y,z
  # imu_body_raw_to_applanix = np.array([0, -1, 0, -1, 0, 0, 0, 0, -1]).reshape(3, 3)
  yfwd2xfwd = np.array([0, 1, 0, -1, 0, 0, 0, 0, 1]).reshape(3, 3)
  # raw_to_robot = yfwd2xfwd @ imu_body_raw_to_applanix
  raw_to_robot = yfwd2xfwd
  # raw to robot
  data[:, 1:4] = data[:, 1:4] @ raw_to_robot.T
  data[:, 4:] = data[:, 4:] @ raw_to_robot.T

  Rw = np.sqrt(np.std(data[:args.N, 1:4], axis=0)).squeeze()
  print('Rw: {}'.format(Rw))

  Ra = np.sqrt(np.std(data[:args.N, 4:], axis=0)).squeeze()
  print('Ra: {}'.format(Ra))

  # TODO: load in Poses, Velocities, Accelerations from SBET

  errs = []
  dt = 1.0 / SBET_RATE
  Ak = get_tran_wnoj(dt)
  Delta_k_inv = get_qinv_wnoj(dt, 1)
  # for seed in tqdm(range(100000, 100100)):
  # TODO: use multiple training sequences to get better estimate of Qc
  # for _ in data:
  #     states = simulator.forward(seed)[0]
  e = []
  for i in range(states.shape[0] - 1):
      e.append(states[i + 1, :].reshape(3, 1) - Ak @ states[i, :].reshape(3, 1))
  errs += e

  qc = 0
  s = 0
  for e in errs:
      s += e.T @ Delta_k_inv @ e
  qc = s / (3 * len(errs))
  print('WNOJ qc trained on data:')
  print(qc)
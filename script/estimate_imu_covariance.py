import numpy as np
import os
import os.path as osp
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, help='path to boreas-2021-...', default='/workspace/raid/krb/boreas/boreas-2021-09-02-11-42')
  parser.add_argument('--N', type=int, default=1000, help='number of IMU measurements to use from the beginning of the trajectory when the vehicle is stationary')
  args = parser.parse_args()

  with open(osp.join(args.root, 'applanix', 'imu.csv')) as f:
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
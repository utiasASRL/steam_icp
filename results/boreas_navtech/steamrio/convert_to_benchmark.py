import os
import os.path as osp
import numpy as np
import numpy.linalg as npla

pose_files = sorted([f for f in os.listdir('.') if 'poses' in f])
print(pose_files)
zup2zdown = np.array([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]], dtype=np.float64)

for pose_file in pose_files:
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    sequence = pose_file.split('_')[0]
    print(sequence)
    gt_file = osp.join('/workspace/raid/krb/boreas', sequence, 'applanix/radar_poses.csv')
    with open(gt_file, 'r') as f:
        h = f.readline()
        gtlines = f.readlines()
    assert len(lines) == len(gtlines)
    new_file = sequence + '.txt'
    with open(new_file, 'w') as f:
        for l, gtl, in zip(lines, gtlines):
            t = gtl.split(',')[0]
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array([float(x) for x in l.split()]).reshape(3, 4)
            T = zup2zdown @ npla.inv(T) @ zup2zdown
            T = [str(x) for x in T.reshape(-1, 1).squeeze().tolist()[:12]]
            outl = ' '.join(T)[:-1]
            lnew = t + ' ' + outl + '\n'
            f.write(lnew)

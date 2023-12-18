import open3d as o3d
from matplotlib import cm
import numpy as np
import numpy.linalg as npla
import matplotlib
import matplotlib.pyplot as plt
import os
import os.path as osp

# root = '/home/krb/ASRL/temp/steam_icp/boreas_velodyne/steamlo'

root = '/home/krb/ASRL/temp/steam_icp/boreas_navtech/steamrio'
os.listdir(root)
maps = sorted([f for f in os.listdir(root) if 'map' in f])
print(maps)

cmap = cm.gist_rainbow
# cmap = cm.winter

points = []
intensity = []

for map in maps:
  pnts = np.loadtxt(osp.join(root, map))
  points.append(pnts[:, :3])
  intensity.append(pnts[:, 3])

points = np.vstack(points)
intensity = np.hstack(intensity)
print(points.shape)
print(intensity.shape)
print('intensity min: {} max: {}'.format(np.min(intensity), np.max(intensity)))

# vmin = -2
# vmax = 10
# vmin = 0
# vmax = 30
vmin = 0.09
vmax = 0.50

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
intensity = np.clip(intensity, vmin, vmax)
colors = cmap((intensity - vmin) / (vmax - vmin))[:, 0:3]
# colors = cmap(intensity)[:, 0:3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
#                                                     std_ratio=2.0)
cl, ind = pcd.remove_radius_outlier(nb_points=2, radius=0.25)
pcd = pcd.select_by_index(ind)

out = np.hstack((points[ind], intensity[ind].reshape(-1, 1)))

np.savetxt('/home/krb/ASRL/temp/steam_icp/boreas_navtech/steamrio/map.txt', out)

vis = o3d.visualization.Visualizer()
vis.create_window(width=3840, height=2160)
render = vis.get_render_option()
render.load_from_json("render_option2.json")

vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

# ctr = vis.get_view_control()
# ctr.convert_from_pinhole_camera_parameters(
#     o3d.io.read_pinhole_camera_parameters('view_option2.json'))

# o3dimg = vis.capture_screen_float_buffer(do_render=True)

# out = np.array(o3dimg)
# vis.destroy_window()
# plt.imshow(out)
# plt.show()

from scipy.spatial.transform import Rotation as R
import numpy as np
from math import acos
from ..mmdet.datasets.kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, mesh_point_to_bbox, euler_angles_to_rotation_matrix

ea1 = np.array([0.14800601, 0.00555563, np.pi])
ea2 = np.array([0.14800601, 0.00555563, -np.pi])

q1 = euler_angles_to_quaternions(ea1)
q2 = euler_angles_to_quaternions(ea2)


q1R = R.from_quat(q1)
q2R = R.from_quat(q2)

# q1R = R.from_euler('YXZ', ea1)
# q2R = R.from_euler('YXZ', ea2)


q2R_inv = R.inv(q2R)
diff = q1R * q2R_inv
W = np.clip(diff.as_quat()[-1], -1. ,1)

W = (acos(W) * 360) / np.pi
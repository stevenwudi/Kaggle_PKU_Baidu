#!/usr/bin/env python
import os
import cv2
import numpy as np
from mmcv import imwrite
import json

# mAP calculation import
from math import acos, pi
from scipy.spatial.transform import Rotation as R
from tools.evaluations.map_calculation import TranslationDistance, RotationDistance, thres_tr_list, thres_ro_list
from sklearn.metrics import average_precision_score

from mmdet.datasets.kaggle_pku_utils import euler_to_Rot
from demo.visualisation_utils import visual_PnP

# Camera internals
camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
print("Camera Matrix :\n {0}".format(camera_matrix))

# Load the car model here
data_root = '/data/Kaggle/pku-autonomous-driving/'
apollo_data_root = '/data/Kaggle/ApolloScape_3D_car/train/'

# Now we have only one 3D keypoint association
car_name = 'baoshijie-kayan.json'

car_model_dict = {}
with open(os.path.join(data_root, 'car_models_json', car_name)) as json_file:
    car_model_dict[car_name[:-5]] = json.load(json_file)

vertices = np.array(car_model_dict[car_name[:-5]]['vertices'])
vertices[:, 1] = -vertices[:, 1]
triangles = np.array(car_model_dict[car_name[:-5]]['faces']) - 1

kp_index = np.array(
    [2651, 2620, 3770, 3811, 3745, 3582, 3951, 4314, 2891, 3820, 3936, 3219, 3846, 4134, 4254, 4247, 3470, 4133, 4234,
     4290, 4273, 3898, 3654, 3800, 2865, 2635, 2599, 2529, 3342, 1157, 2087, 2005, 1973, 1986, 1347, 1155, 686, 356,
     390, 528, 492, 1200, 460, 527, 342, 783, 1406, 540, 811, 1761, 326, 769, 1133, 889, 810, 945, 1954, 1974, 3389,
     2078, 2114, 2824, 2508, 2121, 2134, 2483])

im_all = os.listdir(apollo_data_root + 'keypoints')
# Read Image, we read only one
im_name = im_all[0]
im = cv2.imread(os.path.join(apollo_data_root + 'images', im_name + '.jpg'))
im_combined = im.copy()
size = im.shape


ke_dir = os.path.join(apollo_data_root + 'keypoints')
PnP_pred = []

for kpfile in sorted(os.listdir(os.path.join(ke_dir, im_name))):
    # Read kp file from the ground truth
    kp_txt = os.path.join(ke_dir, im_name, kpfile)
    kp = np.array([x.rstrip().split('\t') for x in open(kp_txt).readlines()])

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([np.array([float(x[1]), float(x[2])]) for x in kp])

    # 3D model points.
    model_points = np.array([vertices[kp_index[int(x[0])]] for x in kp])

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    #  DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.
    if len(image_points) < 6:
        # We only draw GT
        for kp in kp:
            cv2.putText(im, str(int(kp[0])), (int(float(kp[1])), int(float(kp[2]))), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (0, 255, 0))
            cv2.circle(im, (int(float(kp[1])), int(float(kp[2]))), 5, (0, 255, 0), -1)

    else:
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                      image_points,
                                                                      camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Write to prediction
        yaw, pitch, roll = rotation_vector
        yaw, pitch, roll = -pitch, -yaw, -roll
        yaw = np.pi - yaw
        eular_angle = np.array([yaw[0], pitch[0], roll[0]])
        # Note the the y-axis for OpenCV is pointing down, whereas for ApolloScape, the y-axis is pointing up
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
        #eular_angle[0] = np.pi - eular_angle[0]
        translation = - np.array(translation_vector.squeeze())
        PnP_pred.append({'pose': np.concatenate((eular_angle, translation))})
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (projected_point2D, jacobian) = cv2.projectPoints(model_points, rotation_vector, translation_vector,
                                                          camera_matrix, dist_coeffs)

        for i, kp_i in enumerate(projected_point2D):
            cv2.putText(im, str(int(kp[i][0])), (int(float(kp_i[0][0])), int(float(kp_i[0][1]))), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 0, 255))
            cv2.circle(im, (int(float(kp_i[0][0])), int(float(kp_i[0][1]))), 5, (0, 0, 255), -1)
        # We only draw GT
        for kp in kp:
            cv2.putText(im, str(int(kp[0])), (int(float(kp[1])), int(float(kp[2]))), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (0, 255, 0))
            cv2.circle(im, (int(float(kp[1])), int(float(kp[2]))), 5, (0, 255, 0), -1)

# Display image
imwrite(im, '/data/Kaggle/wudi_data/'+im_name+'_PnP.jpg')

### Now we calculate the mAP
with open(os.path.join(apollo_data_root + 'car_poses', im_name+'.json')) as json_file:
    gt_RT = json.load(json_file)

# p = {}
# g = {}
# p['x'], p['y'], p['z'] = translation
# g['x'], g['y'], g['z'] = gt_RT[0]['pose'][3:]
# p['pitch'], p['yaw'], p['roll'] = eular_angle
# g['pitch'], g['yaw'], g['roll'] = gt_RT[0]['pose'][:3]
# translation_diff = TranslationDistance(p, g)
# rotation_diff = RotationDistance(p, g)
# print("Translation distance: %.4f, Rotation distance: %.4f" % (translation_diff, rotation_diff))

for pcar in PnP_pred:
    pcar['x'], pcar['y'], pcar['z'] = pcar['pose'][3:]
    pcar['yaw'], pcar['pitch'], pcar['roll'] = pcar['pose'][:3]


im_combined = visual_PnP(im_combined, PnP_pred, camera_matrix, vertices, triangles)
imwrite(im_combined, '/data/Kaggle/wudi_data/'+im_name+'_combined_PnP.jpg')

ap_list = []


for idx in range(10):
    MAX_VAL = 10 ** 10
    keep_gt = False
    scores = []
    result_flg = []  # 1 for TP, 0 for FP
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]
    n_gt = len(gt_RT)
    for pcar in PnP_pred:
        # find nearest GT
        min_tr_dist = MAX_VAL
        min_idx = -1
        pcar['x'], pcar['y'], pcar['z'] = pcar['pose'][3:]
        pcar['pitch'], pcar['yaw'], pcar['roll'] = pcar['pose'][:3]
        for idx, gcar in enumerate(gt_RT):
            gcar['x'], gcar['y'], gcar['z'] = gcar['pose'][3:]
            gcar['pitch'], gcar['yaw'], gcar['roll'] = gcar['pose'][:3]
            tr_dist = TranslationDistance(pcar, gcar)
            if tr_dist < min_tr_dist:
                min_tr_dist = tr_dist
                min_ro_dist = RotationDistance(pcar, gcar)
                min_idx = idx

        # set the result
        #if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
        if min_tr_dist < thre_tr_dist:
            if not keep_gt:
                gt_RT.pop(min_idx)
            result_flg.append(1)
        else:
            result_flg.append(0)
        scores.append(1.0)

    if np.sum(result_flg) > 0:
        n_tp = np.sum(result_flg)
        recall = n_tp / n_gt
        ap = average_precision_score(result_flg, scores) * recall
    else:
        ap = 0
    ap_list.append(ap)

print('mAP is %.4f.' % np.array(ap_list).mean())




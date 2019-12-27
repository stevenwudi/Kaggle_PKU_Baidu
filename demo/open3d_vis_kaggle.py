"""Perform inference on one or more datasets."""

import argparse
import numpy as np
import os
from collections import OrderedDict
import logging
from tqdm import tqdm
import json
import cv2
import pickle as pkl
from car_models import car_id2name
import open3d as o3d
import pandas as pd
# from open3d import draw_geometries, Vector3dVector, Vector3iVector, TriangleMesh, PointCloud, create_mesh_coordinate_frame
import math
from matplotlib import pyplot as plt
from math import sin, cos

from map_calculation import check_match_single_car, str2coords


def quaternion_to_euler_angle(q):
    """
    Convert quaternion to euler angel.
    该公式适用的yaw, pitch, roll与label里的定义不一样，需要做相应的变换 yaw, pitch, roll => pitch, yaw, roll

    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [yaw, pitch, roll]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    # transform label RPY: yaw, pitch, roll => pitch, yaw, roll
    return pitch, yaw, roll


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])

    return np.dot(Y, np.dot(P, R))


def load_car_models(car_model_dir):
    car_model_dict = {}
    for car_name in tqdm(os.listdir(car_model_dir)):
        with open(os.path.join(car_model_dir, car_name)) as json_file:
            car_model_dict[car_name[:-5]] = json.load(json_file)

    return car_model_dict


def update_road_surface(xmin, xmax, ymin, ymax, zmin, zmax, vertices):
    xmin = min(xmin, vertices[0, :].min())
    xmax = max(xmax, vertices[0, :].max())

    ymin = min(ymin, vertices[1, :].min())
    ymax = max(ymax, vertices[1, :].max())

    zmin = min(zmin, vertices[2, :].min())
    zmax = max(zmax, vertices[2, :].max())

    return xmin, xmax, ymin, ymax, zmin, zmax


def get_road_surface_xyz(xmin, xmax, ymin, ymax, zmin, zmax):
    x = np.linspace(xmin, xmax, int((xmax - xmin) / 0.3))
    z = np.linspace(zmin, zmax, int((zmax - zmin) / 0.3))
    mesh_x, mesh_z = np.meshgrid(x, z)
    mesh_y = np.ones(np.size(mesh_x)) * ymax
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(mesh_z, -1)
    return xyz


def get_open3d_mesh(euler_angle, trans_pred_world, car_model,
                    xmin, xmax, ymin, ymax, zmin, zmax,
                    difficulty_idx):
    yaw, pitch, roll = euler_angle
    yaw, pitch, roll = -pitch, -yaw, -roll
    R = euler_to_Rot(yaw, pitch, roll).T
    T = trans_pred_world
    vertices = np.array(car_model['vertices'])
    vertices[:, 1] = -vertices[:, 1]
    triangles = np.array(car_model['faces']) - 1
    vertices = np.matmul(R, vertices.T) + np.asarray(T)[:, None]

    xmin, xmax, ymin, ymax, zmin, zmax = update_road_surface(xmin, xmax, ymin, ymax, zmin, zmax, vertices)

    mesh_car = o3d.geometry.TriangleMesh()
    mesh_car.vertices = o3d.utility.Vector3dVector(vertices.T)
    mesh_car.triangles = o3d.utility.Vector3iVector(triangles)
    # Computing normal
    mesh_car.compute_vertex_normals()
    # This is a match, RGB-->  The greener, the more positive
    if difficulty_idx >= 0:
        if difficulty_idx <=9:
            print("Matched difficutl level: %d" % difficulty_idx)
            red_color = (10 - difficulty_idx) / 10
            green_color = (difficulty_idx + 1) / 10
            mesh_car.paint_uniform_color([red_color, green_color, 0])
        elif difficulty_idx<=19:
            print("Matched difficutl level: %d" % (difficulty_idx-10))
            blue_color = (20 - difficulty_idx) / 10
            green_color = (difficulty_idx-10 + 1) / 10
            mesh_car.paint_uniform_color([0, green_color, blue_color])
    elif difficulty_idx == -1:
        # Pure red is a FP
        print("False Positive " + str(T))
        mesh_car.paint_uniform_color([1, 0, 0])
    elif difficulty_idx == -2:
        print("False Negative " + str(T))
        mesh_car.paint_uniform_color([1, 1, 1])
    return mesh_car, xmin, xmax, ymin, ymax, zmin, zmax


def open_3d_vis(start_vis_index, valid_pred, train_df, train_img_dir, car_model_dir):
    """
    http://www.open3d.org/docs/tutorial/Basic/visualization.html
    -- Mouse view control --
      Left button + drag        : Rotate.
      Ctrl + left button + drag : Translate.
      Wheel                     : Zoom in/out.
    -- Keyboard view control --
      [/]          : Increase/decrease field of view.
      R            : Reset view point.
      Ctrl/Cmd + C : Copy current view status into the clipboard. (A nice view has been saved as utilites/view.json
      Ctrl/Cmd + V : Paste view status from clipboard.
    -- General control --
      Q, Esc       : Exit window.
      H            : Print help message.
      P, PrtScn    : Take a screen capture.
      D            : Take a depth capture.
      O            : Take a capture of current rendering settings.
    """
    car_models = load_car_models(car_model_dir)
    unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                       19, 20, 23, 25, 27, 28, 31, 32,
                       35, 37, 40, 43, 46, 47, 48, 50,
                       51, 54, 56, 60, 61, 66, 70, 71, 76]
    car_cls_coco = 2
    train_dict = {imgID: str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID, s in
                  zip(train_df['ImageId'], train_df['PredictionString'])}

    for pred_idx in range(len(valid_pred[start_vis_index:])):
        pred = valid_pred[pred_idx + start_vis_index]
        img_name = pred[2]['file_name'].split('/')[-1][:-4]
        print('Visualising: %s' % (str(pred_idx + start_vis_index) + "__" + img_name))
        img = cv2.imread(os.path.join(train_img_dir, img_name + '.jpg'))
        # plt.figure()
        plt.ion()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(str(pred_idx + start_vis_index) + "__" + img_name)
        plt.draw()
        plt.show()
        # cv2.namedWindow(img_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(img_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow(img_name, img)
        # Wudi load the preset view point here, this view is customised by Wudi for Kaggle
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # ctr = vis.get_view_control()
        # param = o3d.io.read_pinhole_camera_parameters('Kaggle_view.json')

        bboxes, segms, six_dof = pred[0], pred[1], pred[2]
        car_cls_score_pred = six_dof['car_cls_score_pred']
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        kaggle_car_labels = [unique_car_mode[x] for x in car_labels]
        car_names = [car_id2name[x].name for x in kaggle_car_labels]
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world']
        euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
        assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
               == len(trans_pred_world) == len(euler_angle) == len(car_names)

        keep_gt = False
        # We also save road surface
        xmin, xmax, ymin, ymax, zmin, zmax = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf
        for car_idx in range(trans_pred_world.shape[0]):
            if bboxes[car_cls_coco][car_idx, -1] > 0.1:

                car_model = car_models[car_names[car_idx]]
                # Now we check whether this car belongs to a TP:
                difficulty_idx, matched_id = check_match_single_car(euler_angle[car_idx], trans_pred_world[car_idx],
                                                                    train_dict[img_name])
                if not keep_gt and matched_id != -1:
                    train_dict[img_name].pop(matched_id)
                # now we draw mesh
                mesh_car, xmin, xmax, ymin, ymax, zmin, zmax = get_open3d_mesh(euler_angle[car_idx],
                                                                               trans_pred_world[car_idx],
                                                                               car_model,
                                                                               xmin, xmax, ymin, ymax, zmin, zmax,
                                                                               difficulty_idx)

                vis.add_geometry(mesh_car)

        for missing_car_idx in train_dict[img_name]:
            euler_angle = np.array([missing_car_idx['pitch'], missing_car_idx['yaw'], missing_car_idx['roll']])
            trans_pred_world = np.array([missing_car_idx['x'], missing_car_idx['y'], missing_car_idx['z']])
            mesh_car, xmin, xmax, ymin, ymax, zmin, zmax = get_open3d_mesh(euler_angle, trans_pred_world, car_model,
                                                                           xmin, xmax, ymin, ymax, zmin, zmax,
                                                                           -2)

            vis.add_geometry(mesh_car)
        # draw mesh frame
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)
        # draw road surface
        xyz = get_road_surface_xyz(xmin, xmax, ymin, ymax, zmin, zmax)
        # Pass xyz to Open3D.PointCloud and visualize
        pcd_road = o3d.geometry.PointCloud()
        pcd_road.points = o3d.utility.Vector3dVector(xyz)
        pcd_road.paint_uniform_color([0, 0, 1])
        vis.add_geometry(pcd_road)

        # Actually drawing of Open3D
        # ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        vis.destroy_window()
        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('Kaggle_view.json', param)

        # o3d.visualization.draw_geometries(mesh_car_all)
        # We draw the RGB image here


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--idx', default=250, type=int, help='starting index for viewing')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    car_model_dir = 'E:\DATASET\pku-autonomous-driving\car_models_json'
    valid_pred_file = r'E:\DATASET\pku-autonomous-driving\cwx_data\validation_all_yihao069e100s5070_resume92Dec24-08-50-226141a3d1.pkl'
    train_img_dir = r'E:\DATASET\pku-autonomous-driving\cwx_data\all_yihao069e100s5070_resume55Dec23-09-14-266141a3d1_valid_ep64_mes_vis'
    train_df = pd.read_csv(r'E:\DATASET\pku-autonomous-driving/train.csv')

    valid_pred = pkl.load(open(valid_pred_file, "rb"))
    args = parse_args()
    start_vis_index = args.idx

    open_3d_vis(start_vis_index=start_vis_index,
                valid_pred=valid_pred,
                train_df=train_df,
                train_img_dir=train_img_dir,
                car_model_dir=car_model_dir)

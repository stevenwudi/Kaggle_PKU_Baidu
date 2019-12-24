"""Perform inference on one or more datasets."""

import argparse
import numpy as np
import os
from collections import OrderedDict
import logging
from utilities import car_models as car_models_all
import pickle as pkl
import _init_paths  # pylint: disable=unused-import
from utilities.eval_car_instances_return_TP import Detect3DEval
from utilities.utils import euler_angles_to_rotation_matrix
from open3d import draw_geometries, Vector3dVector, Vector3iVector, TriangleMesh, PointCloud, create_mesh_coordinate_frame


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', default='ApolloScape', help='Dataset to use')
    parser.add_argument('--dataset_dir', default=r'E:\Thunder\train\\')
    parser.add_argument('--output_dir',
                        default=r'D:\Github\ApolloScape_InstanceSeg\Outputs\e2e_3d_car_101_FPN_triple_head\Sep09-23-42-21_N606-TITAN32_step')
    parser.add_argument('--list_flag', default='val', help='Choosing between [val, test]')
    parser.add_argument('--iou_ignore_threshold', default=1.0, help='Filter out by this iou')
    parser.add_argument('--vis_num', default=50, help='Choosing which image to view')
    parser.add_argument('--criterion_num', default=0, help='[0,1,2,...9]')
    parser.add_argument('--dtScores', default=0.5, help='Detection Score for visualisation')

    return parser.parse_args()


def load_car_models(car_model_dir):
    """Load all the car models
    """
    car_models = OrderedDict([])
    logging.info('loading %d car models' % len(car_models_all.models))
    for model in car_models_all.models:
        car_model = '%s\%s.pkl' % (car_model_dir, model.name)
        # This is a python 3 compatibility
        car_models[model.id] = pkl.load(open(car_model, "rb"), encoding='latin1')
        # fix the inconsistency between obj and pkl
        car_models[model.id]['vertices'][:, [0, 1]] *= -1
        car_models[model.id]['name'] = model.name

        # print("Vertice number is: %d" % len(self.car_models[model.name]['vertices']))
        # print("Face number is: %d" % len(self.car_models[model.name]['faces']))
    return car_models


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


def open_3d_vis(args, output_dir):
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
    json_dir = os.path.join(output_dir, 'json_' + args.list_flag + '_trans') + '_iou' + str(1.0)
    args.test_dir = json_dir
    args.gt_dir = args.dataset_dir + 'car_poses'
    args.res_file = os.path.join(output_dir, 'json_' + args.list_flag + '_res.txt')
    args.simType = None

    car_models = load_car_models(os.path.join(args.dataset_dir, 'car_models'))
    det_3d_metric = Detect3DEval(args)
    evalImgs = det_3d_metric.evaluate()
    image_id = evalImgs['image_id']
    print(image_id)
    # We only draw the most loose constraint here
    gtMatches = evalImgs['gtMatches'][0]
    dtScores = evalImgs['dtScores']
    mesh_car_all = []
    # We also save road surface
    xmin, xmax, ymin, ymax, zmin, zmax = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf
    for car in det_3d_metric._gts[image_id]:
        car_model = car_models[car['car_id']]
        R = euler_angles_to_rotation_matrix(car['pose'][:3])
        vertices = np.matmul(R, car_model['vertices'].T) + np.asarray(car['pose'][3:])[:, None]
        xmin, xmax, ymin, ymax, zmin, zmax = update_road_surface(xmin, xmax, ymin, ymax, zmin, zmax, vertices)

        mesh_car = TriangleMesh()
        mesh_car.vertices = Vector3dVector(vertices.T)
        mesh_car.triangles = Vector3iVector(car_model['faces'] - 1)
        # Computing normal
        mesh_car.compute_vertex_normals()
        # we render the GT car in BLUE
        mesh_car.paint_uniform_color([0, 0, 1])
        mesh_car_all.append(mesh_car)

    for i, car in enumerate(det_3d_metric._dts[image_id]):
        if dtScores[i] > args.dtScores:
            car_model = car_models[car['car_id']]
            R = euler_angles_to_rotation_matrix(car['pose'][:3])
            vertices = np.matmul(R, car_model['vertices'].T) + np.asarray(car['pose'][3:])[:, None]
            mesh_car = TriangleMesh()
            mesh_car.vertices = Vector3dVector(vertices.T)
            mesh_car.triangles = Vector3iVector(car_model['faces'] - 1)
            # Computing normal
            mesh_car.compute_vertex_normals()
            if car['id'] in gtMatches:
                # if it is a true positive, we render it as green
                mesh_car.paint_uniform_color([0, 1, 0])
            else:
                # else we render it as red
                mesh_car.paint_uniform_color([1, 0, 0])
            mesh_car_all.append(mesh_car)

            # Draw the road surface here
            # x = np.linspace(xmin, xmax, 100)
            #  every 0.1 meter

    xyz = get_road_surface_xyz(xmin, xmax, ymin, ymax, zmin, zmax)
    # Pass xyz to Open3D.PointCloud and visualize
    pcd_road = PointCloud()
    pcd_road.points = Vector3dVector(xyz)
    pcd_road.paint_uniform_color([0, 0, 1])
    mesh_car_all.append(pcd_road)

    # draw mesh frame
    mesh_frame = create_mesh_coordinate_frame(size=3, origin=[0, 0, 0])
    mesh_car_all.append(mesh_frame)

    draw_geometries(mesh_car_all)
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Wudi hard coded the following range
    if args.list_flag == 'test':
        args.range = [0, 1041]
    elif args.list_flag == 'val':
        args.range = [0, 206]
    elif args.list_flag == 'train':
        args.range = [0, 3888]

    open_3d_vis(args, output_dir=args.output_dir)
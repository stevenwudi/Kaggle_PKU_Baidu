import os
import cv2
from PIL import Image
import math
import json
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from mmcv.image import imread, imwrite
import torch

from .registry import DATASETS
from .custom import CustomDataset
import pycocotools.mask as maskUtils

import cv2
from matplotlib import transforms
# from matplotlib.patches import Rectangle, Circle, Polygon
# from matplotlib.lines import Line2D
#


from .car_models import car_id2name
from .kitti_utils import bbox_corners, perspective

KITTI_CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                     'Cyclist', 'Tram', 'Misc', 'DontCare']

ObjectData = namedtuple('ObjectData', ['classname', 'truncated', 'occluded', 'alpha', 'bbox',
                                       'position', 'dimensions', 'angle', 'score'])


@DATASETS.register_module
class KittiObjectDataset(CustomDataset):

    def load_annotations(self, data_root, vis=True, vis_dir='/data/KITTI/wudi_data/train_vis'):

        # Some ApolloScape legacy
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        self.car_model_dict = self.load_car_models()

        # Get the root directory containing object detection data
        kitti_split = 'testing' if self.test_mode else 'training'
        self.root = os.path.join(self.data_root, kitti_split)

        # Read split indices from file
        split_file = os.path.dirname(__file__) + '/kitti3d_splits/{}.txt'.format(kitti_split)
        self.indices = self.read_split(split_file)
        annotations = []

        for idx in tqdm(self.indices):
            filename = os.path.join(self.root, 'image_2/{:06d}.png'.format(idx))
            anno = {'filename': filename}
            if not self.test_mode:
                # Load calibration matrix
                calib_file = os.path.join(self.root, 'calib/{:06d}.txt'.format(idx))
                calib = self.read_kitti_calib(calib_file)

                # Load annotations
                label_file = os.path.join(self.root, 'label_2/{:06d}.txt'.format(idx))
                objects = self.read_kitti_objects(label_file)

                anno['calib'] = calib
                anno['objects'] = objects

                if vis:
                    image = imread(anno['filename'])
                    im_save_name = os.path.join(vis_dir, anno['filename'].split('/')[-1])
                    self.visualize_objects(image, calib, objects, im_save_name)

            annotations.append(anno)
        return annotations

    def __len__(self):
        return len(self.img_infos)

    def load_car_models(self, json_model_dir='/data/Kaggle/pku-autonomous-driving'):
        car_model_dir = os.path.join(json_model_dir, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(self.outdir, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict

    def load_anno_idx(self, index):
        idx = self.indices[index]

        # Load image
        img_file = os.path.join(self.root, 'image_2/{:06d}.png'.format(idx))
        image = Image.open(img_file)

        # Load calibration matrix
        calib_file = os.path.join(self.root, 'calib/{:06d}.txt'.format(idx))
        calib = self.read_kitti_calib(calib_file)

        # Load annotations
        label_file = os.path.join(self.root, 'label_2/{:06d}.txt'.format(idx))
        objects = self.read_kitti_objects(label_file)

        return idx, image, calib, objects

    def read_split(self, filename):
        """
        Read a list of indices to a subset of the KITTI training or testing sets
        """
        with open(filename) as f:
            return [int(val) for val in f]

    def read_kitti_calib(self, filename):
        """Read the camera 2 calibration matrix from a text file"""

        with open(filename) as f:
            for line in f:
                data = line.split(' ')
                if data[0] == 'P2:':
                    calib = torch.tensor([float(x) for x in data[1:13]])
                    return calib.view(3, 4)

        raise Exception(
            'Could not find entry for P2 in calib file {}'.format(filename))

    def read_kitti_objects(self, filename):
        objects = list()
        with open(filename, 'r') as fp:

            # Each line represents a single object
            for line in fp:
                objdata = line.split(' ')
                if not (14 <= len(objdata) <= 15):
                    raise IOError('Invalid KITTI object file {}'.format(filename))

                # Parse object data
                objects.append(ObjectData(
                    classname=objdata[0],
                    truncated=objdata[1],
                    occluded=objdata[2],
                    alpha=objdata[3],
                    bbox=[float(p) for p in objdata[4:8]],
                    dimensions=[
                        float(objdata[10]), float(objdata[8]), float(objdata[9])],
                    position=[float(p) for p in objdata[11:14]],
                    angle=float(objdata[14]),
                    score=float(objdata[15]) if len(objdata) == 16 else 1.
                ))
        return objects

    def visualise_pred(self, outputs, args):
        car_cls_coco = 2

        for idx in tqdm(range(len(self.img_infos))):
            ann = self.img_infos[idx]
            img_name = ann['filename']
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                output = outputs[idx]
                # output is a tuple of three elements
                bboxes, segms, six_dof = output[0], output[1], output[2]
                car_cls_score_pred = six_dof['car_cls_score_pred']

                # there could be zero cars
                if len(car_cls_score_pred):
                    car_labels = np.argmax(car_cls_score_pred, axis=1)
                    kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
                    car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

                    assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                           == len(car_names)
                    # now we start to plot the image from kaggle
                    image = self.visualise_box_mask(image, bboxes[car_cls_coco], segms[car_cls_coco], car_names)
                imwrite(image, os.path.join(args.out[:-4] + '_mes_vis/' + img_name.split('/')[-1]))

    def visualise_box_mask(self, img, bboxes, segms, class_names,
                           score_thr=0.1,
                           thickness=1,
                           font_scale=0.8,
                           transparency=0.8):
        """
        This will only draw the bounding box and the mask
        Args:
            img:
            bboxes:
            segms:
            class_names:
            score_thr:
            thickness:
            font_scale:

        Returns:

        """
        if score_thr > 0:
            inds = bboxes[:, -1] > score_thr
            bboxes = bboxes[inds, :]
            segms = np.array(segms)[inds]
            class_names = class_names[inds]

        for bbox_idx in range(len(bboxes)):
            color_ndarray = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color = tuple([int(i) for i in color_ndarray[0]])
            bbox = bboxes[bbox_idx]

            ## below is the predicted mask
            mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
            img[mask_pred] = img[mask_pred] * (1 - transparency) + color_ndarray * transparency

            label_text = class_names[bbox_idx]
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])

            cv2.rectangle(img, left_top, right_bottom, color, thickness=thickness)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_ITALIC, font_scale, color)
        return img


    def visualize_objects(self, image, calib, objects, im_save_name):
        # Visualize objects
        for i, obj in enumerate(objects):
            # we only draw car
            if obj.classname == 'Car':
                image = self.draw_bbox3d_cv2(obj, calib, image)
                # draw mesh here
                image = self.draw_mesh(obj, calib, image)

        imwrite(image, im_save_name)
        return image

    def draw_bbox3d_cv2(self, obj, calib, img , color=(255, 0, 0)):
        # Get corners of 3D bounding box
        corners = bbox_corners(obj)

        # Project into image coordinates
        img_corners = perspective(calib, corners).numpy()
        img_corners = img_corners.astype(int)
        # Draw polygons
        # Front face
        cv2.polylines(img, [img_corners[[1, 3, 7, 5]].reshape(-1, 1, 2)], True, color=color)
        # Back face
        cv2.polylines(img, [img_corners[[0, 2, 6, 4]].reshape(-1, 1, 2)], True, color=color)
        # Fill bottom plane
        #cv2.fillPoly(img, [img_corners[[4, 5, 7, 6]].reshape(-1, 1, 2)],  color=color)
        cv2.fillPoly(img, [img_corners[[0, 1, 3, 2]].reshape(-1, 1, 2)],  color=color)

        pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
        for p in pairs:
            cv2.line(img, (img_corners[p[0]][0], img_corners[p[0]][1]), (img_corners[p[1]][0], img_corners[p[1]][1]), color=color)
        return img

    def draw_mesh(self, obj, calib, img):
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]



        # Get corners of 3D bounding box
        corners = bbox_corners(obj)

        # Project into image coordinates
        img_corners = perspective(calib, corners).numpy()
        img_corners = img_corners.astype(int)
        # Draw polygons
        # Front face
        cv2.polylines(img, [img_corners[[1, 3, 7, 5]].reshape(-1, 1, 2)], True, color=color)
        # Back face
        cv2.polylines(img, [img_corners[[0, 2, 6, 4]].reshape(-1, 1, 2)], True, color=color)
        # Fill bottom plane
        #cv2.fillPoly(img, [img_corners[[4, 5, 7, 6]].reshape(-1, 1, 2)],  color=color)
        cv2.fillPoly(img, [img_corners[[0, 1, 3, 2]].reshape(-1, 1, 2)],  color=color)

        pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
        for p in pairs:
            cv2.line(img, (img_corners[p[0]][0], img_corners[p[0]][1]), (img_corners[p[1]][0], img_corners[p[1]][1]), color=color)
        return img

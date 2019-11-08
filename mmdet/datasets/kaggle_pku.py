import numpy as np
import pandas as pd
from scipy.stats import norm
import json
import os
from tqdm import tqdm
import cv2
from mmcv.image import imread, imwrite
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
from .car_models import car_id2name
from .kaggle_pku_utils import euler_to_Rot


@DATASETS.register_module
class KaggkePKUDataset(CustomDataset):

    CLASSES = ('car')

    def load_annotations(self, ann_file):

        self.image_shape = (2710, 3384)  # this is generally the case
        # From camera.zip
        self.camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                  [0, 2305.8757, 1354.9849],
                                  [0, 0, 1]], dtype=np.float32)
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        print("Loading Car model files...")
        self.car_model_dict = self.load_car_models()

        annotations = []

        train = pd.read_csv(ann_file)
        self.print_statistics(train)

        # this is wrong but we have limited bandwidth to upload images for the moment
        for idx in range(len(train)):
            annotation = self.load_anno_idx(idx, train)


        return annotations

    def load_car_models(self):
        car_model_dir = os.path.join(self.data_root, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(self.data_root, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict

    def load_anno_idx(self, idx, train, draw=True):


        img_name = self.img_prefix + train['ImageId'].iloc[idx] +'.jpg'
        if os.path.isfile(img_name):
            if draw:
                image = imread(img_name)
                mask_all = np.zeros(image.shape)
                alpha = 0.8  # transparency
            gt = self._str2coords(train['PredictionString'].iloc[idx])
            for gt_pred in gt:
                # rendering the car according to:
                # https://www.kaggle.com/ebouteillon/augmented-reality
                car_name = car_id2name[gt_pred['id']].name
                vertices = np.array(self.car_model_dict[car_name]['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                triangles = np.array(self.car_model_dict[car_name]['faces']) - 1

                # project 3D points to 2d image plane
                yaw, pitch, roll = gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']
                # I think the pitch and yaw should be exchanged
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

                if draw:
                    mask = np.zeros(image.shape)
                    for t in triangles:
                        coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]], dtype=np.int32)
                        cv2.polylines(mask, np.int32([coord]), 1, (0, 255, 0))
                    mask_all += mask
            if draw:
                mask_all = mask_all * 255 / mask_all.max()
                image_overlap = cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)
                imwrite(mask_all, os.path.join('/data/Kaggle/wudi_data/train_iamge_gt_vis',train['ImageId'].iloc[idx] +'_mask.jpg'))

                imwrite(image_overlap, os.path.join('/data/Kaggle/wudi_data/train_iamge_gt_vis',train['ImageId'].iloc[idx] +'.jpg'))

        else:
            #assert "Image file does not exist!"
            return []



    def print_statistics(self, train):
        car_per_image = np.array([len(self._str2coords(s)) for s in train['PredictionString']])
        print('Total images: %d, car num sum: %d, minmin: %d, max: %d, mean: %d' %
              (len(car_per_image), car_per_image.sum(), car_per_image.min(), car_per_image.max(), car_per_image.mean()))
        """
        Total images: 4262, car num sum: 49684, minmin: 1, max: 44, mean: 11
        """
        xs, ys = [], []

        for ps in train['PredictionString']:
            x, y = self._get_img_coords(ps)
            xs += list(x)
            ys += list(y)

        xs, ys = np.array(xs), np.array(ys)
        print("x min: %d, max: %d, mean: %d" % (int(min(xs)), int(max(xs)), int(xs.mean())))
        print("y min: %d, max: %d, mean: %d" % (int(min(ys)), int(max(ys)), int(ys.mean())))
        """
        x min: -851, max: 4116, mean: 1551
        y min: 1482, max: 3427, mean: 1820
        """

        # car points looking from the sky
        xs, ys = [], []
        for ps in train['PredictionString']:
            coords = self._str2coords(ps)
            xs += [c['x'] for c in coords]
            ys += [c['y'] for c in coords]
        xs, ys = np.array(xs), np.array(ys)
        print("x min: %d, max: %d, mean: %d" % (int(min(xs)), int(max(xs)), int(xs.mean())))
        print("y min: %d, max: %d, mean: %d" % (int(min(ys)), int(max(ys)), int(ys.mean())))
        """
        x min: -90, max: 519, mean: -3
        y min: 1, max: 689, mean: 9
        """
        # Next we filter our 99.9% data distribution
        xmin, xmax = -80, 80
        ymin, ymax = 1, 50
        xs_cdf = sum((xs > xmin) * (xs < xmax))
        ys_cdf = sum((ys > ymin) * (ys < ymax))
        xs_ys_cdf = sum((xs > xmin) * (xs < xmax) * (ys > ymin) * (ys <ymax))
        print('X within range (%d, %d) will have cdf of: %.6f, outlier number: %d' % (xmin, xmax, xs_cdf / len(xs), len(xs)-xs_cdf))
        print('Y within range (%d, %d) will have cdf of: %.6f, outlier number: %d' % (ymin, ymax, ys_cdf / len(ys), len(ys)-ys_cdf))
        print('Both will have cdf of: %.6f, outlier number: %d' % (xs_ys_cdf / len(ys), len(ys)- xs_ys_cdf))

        car_models = []
        for ps in train['PredictionString']:
            coords = self._str2coords(ps)
            for car in coords:
                car_models.append(car['id'])

        car_models = np.array(np.hstack(car_models))
        print("Car model: max: %d, min: %d, total: %d" % (car_models.max(), car_models.min(), len(car_models)))
        # Car model: max: 76, min: 2, total: 49684
        print('Unique car models:')
        print(np.unique(car_models))
        # array([2, 6, 7, 8, 9, 12, 14, 16, 18, 19, 20, 23, 25, 27, 28, 31, 32,
        #        35, 37, 40, 43, 46, 47, 48, 50, 51, 54, 56, 60, 61, 66, 70, 71, 76])
        print("Number of unique car models: %d" % len(np.unique(car_models))        )


    def _str2coords(self, s, names=('id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z')):
        """
        Input:
            s: PredictionString (e.g. from train dataframe)
            names: array of what to extract from the string
        Output:
            list of dicts with keys from `names`
        """
        coords = []
        for l in np.array(s.split()).reshape([-1, 7]):
            coords.append(dict(zip(names, l.astype('float'))))
            if 'id' in coords[-1]:
                coords[-1]['id'] = int(coords[-1]['id'])
        return coords

    def _get_img_coords(self, s):
        '''
        Input is a PredictionString (e.g. from train dataframe)
        Output is two arrays:
            xs: x coordinates in the image
            ys: y coordinates in the image
        '''
        coords = self._str2coords(s)
        xs = [c['x'] for c in coords]
        ys = [c['y'] for c in coords]
        zs = [c['z'] for c in coords]
        P = np.array(list(zip(xs, ys, zs))).T
        img_p = np.dot(self.camera_matrix, P).T
        img_p[:, 0] /= img_p[:, 2]
        img_p[:, 1] /= img_p[:, 2]
        img_xs = img_p[:, 0]
        img_ys = img_p[:, 1]
        img_zs = img_p[:, 2]  # z = Distance from the camera
        return img_xs, img_ys


    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

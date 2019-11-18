import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import cv2
from mmcv.image import imread, imwrite
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import measure

from .custom import CustomDataset
from .registry import DATASETS
from .car_models import car_id2name
from .kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, mesh_point_to_bbox, euler_angles_to_rotation_matrix


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        elif isinstance(obj, (bytes)):
            return obj.decode("ascii")
        return json.JSONEncoder.default(self, obj)


@DATASETS.register_module
class KaggkePKUDataset(CustomDataset):

    CLASSES = ('car',)

    def load_annotations(self, ann_file,
                         outdir='/data/Kaggle/wudi_data'):
        # some hard coded parameters
        self.image_shape = (2710, 3384)  # this is generally the case
        self.bottom_half = 1480   # this
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        self.cat2label = {car_model: i for i, car_model in enumerate(self.unique_car_mode)}

        # From camera.zip
        self.camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                  [0, 2305.8757, 1354.9849],
                                  [0, 0, 1]], dtype=np.float32)
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)


        print("Loading Car model files...")
        self.car_model_dict = self.load_car_models()

        train = pd.read_csv(ann_file)
        self.print_statistics(train)

        outfile = os.path.join(outdir, ann_file.split('/')[-1].split('.')[0] + '.json')
        #outfile = os.path.join(outdir, ann_file.split('/')[-1].split('.')[0] + '_no_mask.json')

        if os.path.isfile(outfile):
            annotations = json.load(open(outfile, 'r'))
        else:
            annotations = []
            for idx in tqdm(range(len(train))):
                annotation = self.load_anno_idx(idx, train)
                annotations.append(annotation)
            with open(outfile, 'w') as f:
                json.dump(annotations, f, indent=4, cls=NumpyEncoder)

        self.annotations = annotations
        return annotations

    def load_car_models(self):
        car_model_dir = os.path.join(self.data_root, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(self.data_root, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict

    def load_anno_idx(self, idx, train, draw=True, draw_dir='/data/Kaggle/wudi_data/train_iamge_gt_vis'):

        labels = []
        bboxes = []
        rles = []
        eular_angles = []
        quaternion_semispheres = []
        translations = []

        img_name = self.img_prefix + train['ImageId'].iloc[idx] +'.jpg'
        if not os.path.isfile(img_name):
            assert "Image file does not exist!"
        else:
            if draw:
                image = imread(img_name)
                mask_all = np.zeros(image.shape)
                merged_image = image.copy()
                alpha = 0.8  # transparency

            gt = self._str2coords(train['PredictionString'].iloc[idx])
            for gt_pred in gt:
                eular_angle = np.array([gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']])
                translation = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
                quaternion = euler_angles_to_quaternions(eular_angle)
                quaternion_semisphere = quaternion_upper_hemispher(quaternion)

                labels.append(gt_pred['id'])
                eular_angles.append(eular_angle)
                quaternion_semispheres.append(quaternion_semisphere)
                translations.append(translation)
                # rendering the car according to:
                # https://www.kaggle.com/ebouteillon/augmented-reality

                # car_id2name is from:
                # https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/car_models.py
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

                # project 3D points to 2d image plane
                rot_mat = euler_angles_to_rotation_matrix(eular_angle)
                rvect, _ = cv2.Rodrigues(rot_mat)
                imgpts, jac = cv2.projectPoints(np.array(self.car_model_dict[car_name]['vertices']), rvect,
                                                translation, self.camera_matrix,
                                                distCoeffs=None)

                imgpts = np.int32(imgpts).reshape(-1, 2)
                x1, y1, x2, y2 = imgpts[:, 0].min(), imgpts[:, 1].min(), imgpts[:, 0].max(), imgpts[:, 1].max()
                bboxes.append([x1, y1, x2, y2])

                if draw:
                    # project 3D points to 2d image plane
                    mask_seg = np.zeros(image.shape, dtype=np.uint8)
                    for t in triangles:
                        coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]], dtype=np.int32)
                        # This will draw the mask for segmenation
                        cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
                        #cv2.polylines(mask_seg, np.int32([coord]), 1, (0, 255, 0))

                    mask_all += mask_seg
                    #imwrite(mask_seg, os.path.join('/data/Kaggle/wudi_data/train_iamge_gt_vis','mask_demo.jpg'))

                    # Find mask
                    ground_truth_binary_mask = np.zeros(mask_seg.shape, dtype=np.uint8)
                    ground_truth_binary_mask[mask_seg == 255] = 1
                    if self.bottom_half > 0:  # this indicate w
                        ground_truth_binary_mask = ground_truth_binary_mask[int(self.bottom_half):, :]

                    #x1, x2, y1, y2 = mesh_point_to_bbox(ground_truth_binary_mask)

                    # TODO: problem of masking
                    # Taking a kernel for dilation and erosion,
                    # the kernel size is set at 1/10th of the average width and heigh of the car

                    kernel_size = int(((y2-y1)/2 + (x2-x1)/2) / 10)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    # Following is the code to find mask
                    ground_truth_binary_mask_img = ground_truth_binary_mask.sum(axis=2).astype(np.uint8)
                    ground_truth_binary_mask_img[ground_truth_binary_mask_img>1] = 1
                    ground_truth_binary_mask_img = cv2.dilate(ground_truth_binary_mask_img, kernel, iterations=1)
                    ground_truth_binary_mask_img = cv2.erode(ground_truth_binary_mask_img, kernel, iterations=1)
                    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask_img)
                    encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)

                    rles.append(encoded_ground_truth)
                    #bm = maskUtils.decode(encoded_ground_truth)
            #if draw:
            if False:
                mask_all = mask_all * 255 / mask_all.max()
                cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)
                imwrite(merged_image, os.path.join(draw_dir, train['ImageId'].iloc[idx] +'.jpg'))

            if len(bboxes):
                bboxes = np.array(bboxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                eular_angles = np.array(eular_angles, dtype=np.float32)
                quaternion_semispheres = np.array(quaternion_semispheres, dtype=np.float32)
                translations = np.array(translations, dtype=np.float32)
                assert len(gt) == len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(translations)

                annotation = {
                    'filename': img_name,
                    'width': self.image_shape[1],
                    'height': self.image_shape[0],
                    'bboxes': bboxes,
                    'labels': labels,
                    'eular_angles': eular_angles,
                    'quaternion_semispheres': quaternion_semispheres,
                    'translations': translations,
                    'rles': rles
                }
                return annotation

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
        xs, ys, zs = [], [], []
        for ps in train['PredictionString']:
            coords = self._str2coords(ps)
            xs += [c['x'] for c in coords]
            ys += [c['y'] for c in coords]
            zs += [c['z'] for c in coords]
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        print("x min: %d, max: %d, mean: %d, std: %.3f" % (int(min(xs)), int(max(xs)), int(xs.mean()), xs.std()))
        print("y min: %d, max: %d, mean: %d, std: %.3f" % (int(min(ys)), int(max(ys)), int(ys.mean()), ys.std()))
        print("z min: %d, max: %d, mean: %d, std: %.3f" % (int(min(zs)), int(max(zs)), int(zs.mean()), zs.std()))

        """
        x min: -90, max: 519, mean: -3, std: 14.560
        y min: 1, max: 689, mean: 9, std: 6.826
        z min: 3, max: 3502, mean: 52, std: 40.046
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
        print("Number of unique car models: %d" % len(np.unique(car_models)))
        # 34

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
        ann_info = self.img_infos[idx]
        return self._parse_ann_info(ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
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
        gt_class_labels = []    # this will always be fixed as car class
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        quaternion_semispheres = []
        translations = []

        for i in range(len(ann_info['bboxes'])):
            x1, y1, x2, y2 = ann_info['bboxes'][i]
            w, h = x2-x1, y2-y1
            if w < 1 or h < 1:
                continue
            translation = ann_info['translations'][i]
            # X within range (-80, 80) will have cdf of: 0.999738, outlier number: 13
            # Y within range (1, 50) will have cdf of: 0.999819, outlier number: 9
            if translation[0] < -80 or translation[0] > 80 or translation[1] < 1 or translation[1] > 50:
                continue
            if self.bottom_half:
                # we only take bottom half image
                bbox = [x1, y1 - self.bottom_half, x2, y2 - self.bottom_half]
            else:
                bbox = [x1, y1, x2, y2]
            if ann_info.get('iscrowd', False):   # TODO: train mask need to include
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)

                # There are only 34 car in this training dataset:
                gt_label = self.cat2label[ann_info['labels'][i]]
                gt_labels.append(gt_label)
                gt_class_labels.append(3)  # coco 3 is "car" class
                mask = maskUtils.decode(ann_info['rles'][i])
                gt_masks_ann.append(mask)

                quaternion_semispheres.append(ann_info['quaternion_semispheres'][i])
                translations.append(translation)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            quaternion_semispheres = np.array(quaternion_semispheres, dtype=np.float32)
            translations = np.array(translations, dtype=np.float32)

            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_class_labels,
            carlabels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            quaternion_semispheres=quaternion_semispheres,
            translations=translations)

        return ann

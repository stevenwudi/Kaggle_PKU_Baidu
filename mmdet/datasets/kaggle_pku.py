import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import cv2
from math import sin, cos
from mmcv.image import imread, imwrite
import mmcv
from pycocotools import mask as maskUtils
import multiprocessing
import copy

from .custom import CustomDataset
from .registry import DATASETS
from .car_models import car_id2name

from .kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, quaternion_to_euler_angle, draw_line, draw_points, non_max_suppression_fast

from .visualisation_utils import draw_result_kaggle_pku, draw_box_mesh_kaggle_pku, refine_yaw_and_roll, \
    restore_x_y_from_z_withIOU, get_IOU, nms_with_IOU, nms_with_IOU_and_vote, nms_with_IOU_and_vote_return_index

from albumentations.augmentations import transforms
from math import acos, pi
from scipy.spatial.transform import Rotation as R


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
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        elif isinstance(obj, (bytes)):
            return obj.decode("ascii")
        return json.JSONEncoder.default(self, obj)


@DATASETS.register_module
class KagglePKUDataset(CustomDataset):
    CLASSES = ('car',)

    def load_annotations(self, ann_file, outdir='/data/Kaggle/pku-autonomous-driving'):

        # some hard coded parameters
        self.outdir = outdir
        self.image_shape = (2710, 3384)  # this is generally the case
        self.bottom_half = 1480  # this
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        self.cat2label = {car_model: i for i, car_model in enumerate(self.unique_car_mode)}

        # From camera.zip
        self.camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                       [0, 2305.8757, 1354.9849],
                                       [0, 0, 1]], dtype=np.float32)

        print("Loading Car model files...")
        self.car_model_dict = self.load_car_models()
        self.car_id2name = car_id2name
        annotations = []
        if not self.test_mode:
            outfile = ann_file
            if os.path.isfile(outfile):
                annotations = json.load(open(outfile, 'r'))
            else:
                PATH = '/data/Kaggle/ApolloScape_3D_car/train/split/'
                ImageId = [i.strip() for i in open(PATH + 'train-list.txt').readlines()]
                train = pd.read_csv(ann_file)
                for idx in tqdm(range(len(train))):
                    filename = train['ImageId'].iloc[idx] + '.jpg'
                    if filename not in ImageId:
                        continue
                    annotation = self.load_anno_idx(idx, train)
                    annotations.append(annotation)
                with open(outfile, 'w') as f:
                    json.dump(annotations, f, indent=4, cls=NumpyEncoder)
            annotations = self.clean_corrupted_images(annotations)
            annotations = self.clean_outliers(annotations)

            self.print_statistics_annotations(annotations)

        else:
            if os.path.isfile(ann_file):  # This for evulating ApolloScape
                # Using readlines()
                file = open(ann_file, 'r')
                lines = file.readlines()

                for fn in lines:
                    filename = os.path.join(self.img_prefix, fn.replace('\n', ''))
                    info = {'filename': filename}
                    annotations.append(info)
            else:
                for fn in os.listdir(self.img_prefix):
                    filename = os.path.join(self.img_prefix, fn)
                    info = {'filename': filename}
                    annotations.append(info)

            # We also generate the albumentation enhances valid images
            # below is a hard coded list....
            if False:
                self.generate_albu_valid(annotations)

        # We also will generate a pickle file if the translation, SSD like offset regression
        # this will need to be done only once
        if False:
            self.group_rectangles(annotations)

        self.annotations = annotations

        return annotations

    def load(self, t):
        return self.load_anno_idx(*t)

    def generate_albu_valid(self, annotations):

        num_albu = len(self.pipeline_dict[-1].transforms[-3].transforms)
        for i_albu in range(num_albu):
            params = self.pipeline_dict[-1].transforms[-3].transforms[i_albu]
            # always set p=1 so that we will always tranform the image
            params['p'] = 1
            operation = getattr(transforms, params['type'])
            # delete the 'type'
            params_input = params.copy()
            del params_input['type']

            im_out_dir = self.img_prefix + '_' + params['type']
            if not os.path.exists(im_out_dir):
                os.mkdir(im_out_dir)
            print('Generating: %s' % params['type'])
            for im_idx in tqdm(range(len(annotations))):
                image = imread(annotations[im_idx]['filename'])
                img_aug = operation(**params_input)(image=image)['image']
                im_out_file = annotations[im_idx]['filename'].split('/')[-1]
                imwrite(img_aug, os.path.join(im_out_dir, im_out_file))

    def load_car_models(self):
        car_model_dir = os.path.join(self.outdir, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(self.outdir, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict

    def RotationDistance(self, p, g):
        true = [g[1], g[0], g[2]]
        pred = [p[1], p[0], p[2]]
        q1 = R.from_euler('xyz', true)
        q2 = R.from_euler('xyz', pred)
        diff = R.inv(q2) * q1
        W = np.clip(diff.as_quat()[-1], -1., 1.)

        # in the official metrics code:
        # Peking University/Baidu - Autonomous Driving
        #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
        # this code treat θ and θ+2π differntly.
        # So this should be fixed as follows.
        W = (acos(W) * 360) / pi
        if W > 180:
            W = 360 - W
        return W

    def load_anno_idx(self, idx, train, draw=False, draw_dir='/data/cyh/kaggle/train_image_gt_vis'):

        labels = []
        bboxes = []
        rles = []
        eular_angles = []
        quaternion_semispheres = []
        translations = []

        img_name = self.img_prefix + train['ImageId'].iloc[idx] + '.jpg'
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

                new_eular_angle = quaternion_to_euler_angle(quaternion_semisphere)
                distance = self.RotationDistance(new_eular_angle, eular_angle)
                # distance = np.sum(np.abs(new_eular_angle - eular_angle))
                if distance > 0.001:
                    print("Wrong !!!", img_name)

                labels.append(gt_pred['id'])
                eular_angles.append(eular_angle)
                quaternion_semispheres.append(quaternion_semisphere)
                translations.append(translation)
                # rendering the car according to:
                # Augmented Reality | Kaggle

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
                x1, y1, x2, y2 = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), img_cor_points[:,
                                                                                         0].max(), img_cor_points[:,
                                                                                                   1].max()
                bboxes.append([x1, y1, x2, y2])

                # project 3D points to 2d image plane
                if draw:
                    # project 3D points to 2d image plane
                    mask_seg = np.zeros(image.shape, dtype=np.uint8)
                    mask_seg_mesh = np.zeros(image.shape, dtype=np.uint8)
                    for t in triangles:
                        coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                                         dtype=np.int32)
                        # This will draw the mask for segmenation
                        cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
                        cv2.polylines(mask_seg_mesh, np.int32([coord]), 1, (0, 255, 0))

                    mask_all += mask_seg_mesh

                    ground_truth_binary_mask = np.zeros(mask_seg.shape, dtype=np.uint8)
                    ground_truth_binary_mask[mask_seg == 255] = 1
                    if self.bottom_half > 0:  # this indicate w
                        ground_truth_binary_mask = ground_truth_binary_mask[int(self.bottom_half):, :]

                    # x1, x2, y1, y2 = mesh_point_to_bbox(ground_truth_binary_mask)

                    # TODO: problem of masking
                    # Taking a kernel for dilation and erosion,
                    # the kernel size is set at 1/10th of the average width and heigh of the car

                    kernel_size = int(((y2 - y1) / 2 + (x2 - x1) / 2) / 10)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    # Following is the code to find mask
                    ground_truth_binary_mask_img = ground_truth_binary_mask.sum(axis=2).astype(np.uint8)
                    ground_truth_binary_mask_img[ground_truth_binary_mask_img > 1] = 1
                    ground_truth_binary_mask_img = cv2.dilate(ground_truth_binary_mask_img, kernel, iterations=1)
                    ground_truth_binary_mask_img = cv2.erode(ground_truth_binary_mask_img, kernel, iterations=1)
                    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask_img)
                    encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)

                    rles.append(encoded_ground_truth)
                    # bm = maskUtils.decode(encoded_ground_truth)

            if draw:
                # if False:
                mask_all = mask_all * 255 / mask_all.max()
                cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)

                for box in bboxes:
                    cv2.rectangle(merged_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                                  thickness=5)

                imwrite(merged_image, os.path.join(draw_dir, train['ImageId'].iloc[idx] + '.jpg'))

            if len(bboxes):
                bboxes = np.array(bboxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                eular_angles = np.array(eular_angles, dtype=np.float32)
                quaternion_semispheres = np.array(quaternion_semispheres, dtype=np.float32)
                translations = np.array(translations, dtype=np.float32)
                assert len(gt) == len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(
                    translations)

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

    def eular_angle_classification(self, annotations, draw_dir=""):
        # for ann in tqdm(annotations):
        for ann in tqdm(annotations[5000: 5010]):
            # for ann in tqdm(annotations[0: 50]):

            img_name = ann['filename']
            image = imread(img_name)
            mask_all = np.zeros(image.shape)
            merged_image = image.copy()
            alpha = 0.9  # transparency

            bboxes = ann['bboxes']
            labels = ann['labels']
            eular_angles = ann['eular_angles']
            quaternion_semispheres = ann['quaternion_semispheres']
            translations = ann['translations']
            assert len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(translations)

            for gt_car_idx in range(len(ann['quaternion_semispheres'])):

                eular_angle = np.array(eular_angles[gt_car_idx])

                # if 'Camera' in img_name:  # this is an apolloscape dataset
                #     eular_angle_kaggle = np.array([eular_angle[1], eular_angle[0], eular_angle[2]])
                # elif 'ID' in img_name:
                #     eular_angle_kaggle = eular_angle
                # else:
                #     print("Unidentified class")

                quaternion = euler_angles_to_quaternions(eular_angle)
                quaternion_semisphere = quaternion_upper_hemispher(quaternion)
                ea_make = quaternion_to_euler_angle(quaternion_semisphere)

                json_q = quaternion_semispheres[gt_car_idx]
                ea_json = quaternion_to_euler_angle(json_q)
                ea_json = np.array(ea_json)

                # q1 = R.from_euler('xyz', eular_angle)
                # q2 = R.from_euler('xyz', q)

                # print('GT eular angle: ', eular_angle)
                # print('Generate eular angle:', ea_make)
                # print('Json generated eular angle', ea_json)
                # print('Generate q:', quaternion_semisphere)
                # print('Json q:', json_q)
                # print("diff is: %f" % np.sum(np.abs(ea_json-ea_make)))
                if self.RotationDistance(ea_make, ea_json) > 0.01:
                    print('Wrong!!!!!!!!!!!!!')

                # rendering the car according to:
                # Augmented Reality | Kaggle
                # car_id2name is from:
                # https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/car_models.py
                car_name = car_id2name[labels[gt_car_idx]].name
                vertices = np.array(self.car_model_dict[car_name]['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                triangles = np.array(self.car_model_dict[car_name]['faces']) - 1
                translation = np.array(translations[gt_car_idx])

                Rt = np.eye(4)
                Rt[:3, 3] = translation
                # project 3D points to 2d image plane
                # Apollo below is correct
                # https://en.wikipedia.org/wiki/Euler_angles
                # Y, P, R = euler_to_Rot_YPR(eular_angle[1], eular_angle[0], eular_angle[2])
                rot_mat = euler_to_Rot(eular_angle[0], eular_angle[1], eular_angle[2]).T
                # check eular from rot mat
                Rt[:3, :3] = rot_mat
                Rt = Rt[:3, :]
                P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
                P[:, :-1] = vertices
                P = P.T

                img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
                img_cor_points = img_cor_points.T
                img_cor_points[:, 0] /= img_cor_points[:, 2]
                img_cor_points[:, 1] /= img_cor_points[:, 2]

                x1, y1, x2, y2 = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), img_cor_points[:,
                                                                                         0].max(), img_cor_points[:,
                                                                                                   1].max()
                bboxes.append([x1, y1, x2, y2])

                # project 3D points to 2d image plane
                mask_seg = np.zeros(image.shape, dtype=np.uint8)
                for t in triangles:
                    coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                                     dtype=np.int32)
                    # This will draw the mask for segmenation
                    # cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
                    cv2.polylines(mask_seg, np.int32([coord]), 1, (0, 255, 0))

                mask_all += mask_seg

            mask_all = mask_all * 255 / mask_all.max()
            cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)
            im_write_file = os.path.join(draw_dir, img_name.split('/')[-1])
            print("Writing image to: %s" % os.path.join(draw_dir, img_name.split('/')[-1]))
            imwrite(merged_image, im_write_file)

        return True

    def plot_and_examine(self, annotations, draw_dir='/data/Kaggle/wudi_data/train_image_gt_vis'):

        # for ann in tqdm(annotations):
        # for ann in tqdm(annotations[5000: 5010]):
        for ann in tqdm(annotations):

            img_name = ann['filename']
            image = imread(img_name)
            mask_all = np.zeros(image.shape)
            merged_image = image.copy()
            alpha = 0.9  # transparency

            bboxes = ann['bboxes']
            labels = ann['labels']
            eular_angles = ann['eular_angles']
            quaternion_semispheres = ann['quaternion_semispheres']
            translations = ann['translations']
            assert len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(translations)

            for gt_car_idx in range(len(ann['quaternion_semispheres'])):

                eular_angle = np.array(eular_angles[gt_car_idx])

                # if 'Camera' in img_name:  # this is an apolloscape dataset
                #     eular_angle_kaggle = np.array([eular_angle[1], eular_angle[0], eular_angle[2]])
                # elif 'ID' in img_name:
                #     eular_angle_kaggle = eular_angle
                # else:
                #     print("Unidentified class")

                quaternion = euler_angles_to_quaternions(eular_angle)
                quaternion_semisphere = quaternion_upper_hemispher(quaternion)
                ea_make = quaternion_to_euler_angle(quaternion_semisphere)

                json_q = quaternion_semispheres[gt_car_idx]
                ea_json = quaternion_to_euler_angle(json_q)
                ea_json = np.array(ea_json)

                # q1 = R.from_euler('xyz', eular_angle)
                # q2 = R.from_euler('xyz', q)

                # print('GT eular angle: ', eular_angle)
                # print('Generate eular angle:', ea_make)
                # print('Json generated eular angle', ea_json)
                # print('Generate q:', quaternion_semisphere)
                # print('Json q:', json_q)
                # print("diff is: %f" % np.sum(np.abs(ea_json-ea_make)))
                if self.RotationDistance(ea_make, ea_json) > 0.01:
                    print('Wrong!!!!!!!!!!!!!')

                # rendering the car according to:
                # Augmented Reality | Kaggle
                # car_id2name is from:
                # https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/car_models.py
                car_name = car_id2name[labels[gt_car_idx]].name
                vertices = np.array(self.car_model_dict[car_name]['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                triangles = np.array(self.car_model_dict[car_name]['faces']) - 1
                translation = np.array(translations[gt_car_idx])

                Rt = np.eye(4)
                Rt[:3, 3] = translation
                # project 3D points to 2d image plane
                # Apollo below is correct
                # https://en.wikipedia.org/wiki/Euler_angles
                # Y, P, R = euler_to_Rot_YPR(eular_angle[1], eular_angle[0], eular_angle[2])
                rot_mat = euler_to_Rot(-eular_angle[1], -eular_angle[0], -eular_angle[2]).T
                # check eular from rot mat
                Rt[:3, :3] = rot_mat
                Rt = Rt[:3, :]
                P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
                P[:, :-1] = vertices
                P = P.T

                img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
                img_cor_points = img_cor_points.T
                img_cor_points[:, 0] /= img_cor_points[:, 2]
                img_cor_points[:, 1] /= img_cor_points[:, 2]

                x1, y1, x2, y2 = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), img_cor_points[:,
                                                                                         0].max(), img_cor_points[:,
                                                                                                   1].max()
                bboxes.append([x1, y1, x2, y2])

                # project 3D points to 2d image plane
                mask_seg = np.zeros(image.shape, dtype=np.uint8)
                for t in triangles:
                    coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                                     dtype=np.int32)
                    # This will draw the mask for segmenation
                    # cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
                    cv2.polylines(mask_seg, np.int32([coord]), 1, (0, 255, 0))

                mask_all += mask_seg

            mask_all = mask_all * 255 / mask_all.max()
            cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)
            im_write_file = os.path.join(draw_dir, img_name.split('/')[-1])
            print("Writing image to: %s" % os.path.join(draw_dir, img_name.split('/')[-1]))
            imwrite(merged_image, im_write_file)

        return True

    def visualise_pred_postprocessing(self, outputs, args):
        car_cls_coco = 2

        for idx in tqdm(range(len(outputs))):
            # ann = self.annotations[idx]
            test_folder = '/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/test_images/'
            img_name = os.path.join(test_folder, os.path.basename(outputs[idx][2]['file_name']))

            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                output = outputs[idx]
                # output is a tuple of three elements
                bboxes, segms, six_dof = output[0], output[1], output[2]
                car_cls_score_pred = six_dof['car_cls_score_pred']
                quaternion_pred = six_dof['quaternion_pred']
                trans_pred_world = six_dof['trans_pred_world'].copy()
                euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
                car_labels = np.argmax(car_cls_score_pred, axis=1)
                kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
                car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

                assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                       == len(trans_pred_world) == len(euler_angle) == len(car_names)

                # print('change ',trans_pred_world,trans_pred_world_refined)
                quaternion_semisphere_refined, flag = refine_yaw_and_roll(image, bboxes[car_cls_coco],
                                                                          segms[car_cls_coco], car_names, euler_angle,
                                                                          quaternion_pred, trans_pred_world,
                                                                          self.car_model_dict,
                                                                          self.camera_matrix)
                if flag:
                    output[2]['quaternion_pred'] = quaternion_semisphere_refined
                    euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])

                trans_pred_world_refined = restore_x_y_from_z_withIOU(image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                                      car_names, euler_angle, trans_pred_world,
                                                                      self.car_model_dict,
                                                                      self.camera_matrix)
                output[2]['trans_pred_world'] = trans_pred_world_refined

                # img_box_mesh_refined = self.visualise_box_mesh(image,bboxes[car_cls_coco], segms[car_cls_coco],car_names, euler_angle,trans_pred_world_refined)
                # img_box_mesh_refined, iou_flag = self.visualise_box_mesh(image,bboxes[car_cls_coco], segms[car_cls_coco],car_names, euler_angle,trans_pred_world)
                # if iou_flag:
                #     print('iou problem',os.path.basename(img_name))
                # img_kaggle = self.visualise_kaggle(image, coords)
                # img_mesh = self.visualise_mesh(image, bboxes[car_cls_coco], segms[car_cls_coco], car_names, euler_angle,
                #                                trans_pred_world)
                # imwrite(img_kaggle, os.path.join(args.out[:-4] + '_kaggle_vis/' + img_name.split('/')[-1]))
                # imwrite(img_mesh, os.path.join(args.out[:-4] + '_mes_vis/' + img_name.split('/')[-1]))
                # img_box_mesh_half = cv2.resize(img_box_mesh,None,fx=0.5,fy=0.5)
                # img_kaggle_half = cv2.resize(img_kaggle,None,fx=0.5,fy=0.5)
                # img_concat = np.concatenate([img_kaggle_half,img_box_mesh_half],axis=1)
                # imwrite(img_concat, os.path.join(args.out[:-4] + '_mes_box_vis/' + img_name.split('/')[-1]))
                # imwrite(img_box_mesh, os.path.join(args.out[:-4] + '_mes_box_vis_10_0.7/' + img_name.split('/')[-1]))
                # imwrite(img_box_mesh_refined, os.path.join(args.out[:-4] + '_mes_box_vis_10_0.7_IOU=0/' + img_name.split('/')[-1])[:-4]+'_refined.jpg')

        return outputs

    def distributed_visualise_pred_merge_postprocessing(self, img_id, outputs, args, vote=2, tmp_dir="./results/",
                                                        draw_flag=False):
        car_cls_coco = 2

        bboxes_list = []
        segms_list = []
        six_dof_list = []
        bboxes_with_IOU_list = []

        bboxes_merge = outputs[0][img_id][0].copy()
        segms_merge = outputs[0][img_id][1].copy()
        six_dof_merge = outputs[0][img_id][2].copy()

        last_name = ""
        for i, output in enumerate(outputs):
            a = output[img_id]
            file_name = os.path.basename(a[2]['file_name'])
            if last_name != "" and file_name != last_name:
                assert "Image error!"
            last_name = file_name

            img_name = os.path.join(self.img_prefix, file_name)
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"

            image = imread(img_name)
            bboxes, segms, six_dof = a[0], a[1], a[2]
            bboxes_list.append(bboxes)
            segms_list.append(segms)
            six_dof_list.append(six_dof)

            bboxes_with_IOU = get_IOU(image, bboxes[car_cls_coco], segms[car_cls_coco], six_dof,
                                      car_id2name, self.car_model_dict, self.unique_car_mode, self.camera_matrix)

            new_bboxes_with_IOU = np.zeros((bboxes_with_IOU.shape[0], bboxes_with_IOU.shape[1] + 1))
            for bbox_idx in range(bboxes_with_IOU.shape[0]):
                new_bboxes_with_IOU[bbox_idx] = np.append(bboxes_with_IOU[bbox_idx], float(i))

            bboxes_with_IOU_list.append(new_bboxes_with_IOU)

        bboxes_with_IOU = np.concatenate(bboxes_with_IOU_list, axis=0)
        inds = nms_with_IOU_and_vote(bboxes_with_IOU, vote=vote)  ## IOU nms filter out processing return output indices
        inds = np.array(inds)

        inds_list = []
        start = 0
        for bboxes_iou in bboxes_with_IOU_list:
            end = bboxes_iou.shape[0] + start
            i = np.where((inds >= start) & (inds < end))
            if i:
                inds_current = inds[i] - start
            else:
                inds_current = []
            inds_list.append(inds_current)
            start = end

        bboxes_merge_concat = []
        segms_merge_concat = []
        car_cls_score_pred_concat = []
        quaternion_pred_concat = []
        trans_pred_world_concat = []

        for ids, bboxes, segms, six_dof in zip(inds_list, bboxes_list, segms_list, six_dof_list):
            bboxes_merge_concat.append(bboxes[car_cls_coco][ids])
            segms_merge_concat.append(np.array(segms[car_cls_coco])[ids])
            car_cls_score_pred_concat.append(six_dof['car_cls_score_pred'][ids])
            quaternion_pred_concat.append(six_dof['quaternion_pred'][ids])
            trans_pred_world_concat.append(six_dof['trans_pred_world'][ids])

        bboxes_merge[car_cls_coco] = np.concatenate(bboxes_merge_concat, axis=0)
        segms_merge[car_cls_coco] = np.concatenate(segms_merge_concat, axis=0)
        six_dof_merge['car_cls_score_pred'] = np.concatenate(car_cls_score_pred_concat, axis=0)
        six_dof_merge['quaternion_pred'] = np.concatenate(quaternion_pred_concat, axis=0)
        six_dof_merge['trans_pred_world'] = np.concatenate(trans_pred_world_concat, axis=0)

        output_model_merge = (bboxes_merge, segms_merge, six_dof_merge)

        if draw_flag:
            car_cls_score_pred = six_dof_merge['car_cls_score_pred']
            quaternion_pred = six_dof_merge['quaternion_pred']
            trans_pred_world = six_dof_merge['trans_pred_world'].copy()
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
            car_labels = np.argmax(car_cls_score_pred, axis=1)
            kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
            car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])
            # img_box_mesh_refined = self.visualise_box_mesh(image,bboxes[car_cls_coco], segms[car_cls_coco],car_names, euler_angle,trans_pred_world_refined)
            img_box_mesh_refined, iou_flag = self.visualise_box_mesh(image, bboxes_merge[car_cls_coco],
                                                                     segms_merge[car_cls_coco], car_names,
                                                                     euler_angle, trans_pred_world)
            imwrite(img_box_mesh_refined,
                    os.path.join(args.out[:-4] + '_mes_box_vis_merged/' + img_name.split('/')[-1])[
                    :-4] + '_merged.jpg')

        tmp_file = os.path.join(tmp_dir, "{}.pkl".format(last_name[:-4]))
        mmcv.dump(output_model_merge, tmp_file)
        return output_model_merge

    def distributed_visualise_pred_merge_postprocessing_weight_merge(self, img_id, outputs, args, vote=0,
                                                                     tmp_dir="./results/", draw_flag=False):
        car_cls_coco = 2

        bboxes_list = []
        segms_list = []
        six_dof_list = []
        bboxes_with_IOU_list = []

        bboxes_merge = outputs[0][img_id][0].copy()
        segms_merge = outputs[0][img_id][1].copy()
        six_dof_merge = outputs[0][img_id][2].copy()

        last_name = ""
        if vote == 0:
            vote = len(outputs)
        for i, output in enumerate(outputs):
            a = output[img_id]
            file_name = os.path.basename(a[2]['file_name'])
            if last_name != "" and file_name != last_name:
                assert "Image error!"
            last_name = file_name

            img_name = os.path.join(self.img_prefix, file_name)
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"

            image = imread(img_name)
            bboxes, segms, six_dof = a[0], a[1], a[2]
            bboxes_list.append(bboxes)
            segms_list.append(segms)
            six_dof_list.append(six_dof)

            bboxes_with_IOU = get_IOU(image, bboxes[car_cls_coco], segms[car_cls_coco], six_dof,
                                      car_id2name, self.car_model_dict, self.unique_car_mode, self.camera_matrix)

            new_bboxes_with_IOU = np.zeros((bboxes_with_IOU.shape[0], bboxes_with_IOU.shape[1] + 1))
            for bbox_idx in range(bboxes_with_IOU.shape[0]):
                new_bboxes_with_IOU[bbox_idx] = np.append(bboxes_with_IOU[bbox_idx], float(i))

            bboxes_with_IOU_list.append(new_bboxes_with_IOU)

        bboxes_with_IOU = np.concatenate(bboxes_with_IOU_list, axis=0)
        inds_index = nms_with_IOU_and_vote_return_index(bboxes_with_IOU, vote=vote)  ## IOU nms filter out processing return output indices
        inds = np.array(list(inds_index.keys()))

        trans_pred_world = np.concatenate([sd['trans_pred_world'] for sd in six_dof_list], axis=0)

        # Now we weighted average of the translation
        for ii in inds_index:
            weight = bboxes_with_IOU[:, 5][inds_index[ii]] / np.sum(bboxes_with_IOU[:, 5][inds_index[ii]])
            trans_pred_world[ii] = np.sum(trans_pred_world[inds_index[ii]] * np.expand_dims(weight, axis=1), axis=0)

        inds_list = []
        start = 0
        for bi in range(len(bboxes_with_IOU_list)):
            bboxes_iou = bboxes_with_IOU_list[bi]
            end = bboxes_iou.shape[0] + start

            i = np.where((inds >= start) & (inds < end))
            if i:
                inds_i = inds[i] - start
            else:
                inds_i = []
            six_dof_list[bi]['trans_pred_world'] = trans_pred_world[start:end]
            inds_list.append(inds_i)
            start = end

        bboxes_merge_concat = []
        segms_merge_concat = []
        car_cls_score_pred_concat = []
        quaternion_pred_concat = []
        trans_pred_world_concat = []

        for ids, bboxes, segms, six_dof in zip(inds_list, bboxes_list, segms_list, six_dof_list):
            bboxes_merge_concat.append(bboxes[car_cls_coco][ids])
            segms_merge_concat.append(np.array(segms[car_cls_coco])[ids])
            car_cls_score_pred_concat.append(six_dof['car_cls_score_pred'][ids])
            quaternion_pred_concat.append(six_dof['quaternion_pred'][ids])
            trans_pred_world_concat.append(six_dof['trans_pred_world'][ids])

        bboxes_merge[car_cls_coco] = np.concatenate(bboxes_merge_concat, axis=0)
        segms_merge[car_cls_coco] = np.concatenate(segms_merge_concat, axis=0)
        six_dof_merge['car_cls_score_pred'] = np.concatenate(car_cls_score_pred_concat, axis=0)
        six_dof_merge['quaternion_pred'] = np.concatenate(quaternion_pred_concat, axis=0)
        six_dof_merge['trans_pred_world'] = np.concatenate(trans_pred_world_concat, axis=0)

        output_model_merge = (bboxes_merge, segms_merge, six_dof_merge)

        if draw_flag:
            car_cls_score_pred = six_dof_merge['car_cls_score_pred']
            quaternion_pred = six_dof_merge['quaternion_pred']
            trans_pred_world = six_dof_merge['trans_pred_world'].copy()
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
            car_labels = np.argmax(car_cls_score_pred, axis=1)
            kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
            car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])
            # img_box_mesh_refined = self.visualise_box_mesh(image,bboxes[car_cls_coco], segms[car_cls_coco],car_names, euler_angle,trans_pred_world_refined)
            img_box_mesh_refined, iou_flag = self.visualise_box_mesh(image, bboxes_merge[car_cls_coco],
                                                                     segms_merge[car_cls_coco], car_names,
                                                                     euler_angle, trans_pred_world)
            imwrite(img_box_mesh_refined,
                    os.path.join(args.out[:-4] + '_mes_box_vis_merged/' + img_name.split('/')[-1])[
                    :-4] + '_merged.jpg')

        tmp_file = os.path.join(tmp_dir, "{}.pkl".format(last_name[:-4]))
        mmcv.dump(output_model_merge, tmp_file)
        return output_model_merge

    def visualise_pred_merge_postprocessing(self, outputs, args, conf_thred=0.8):
        car_cls_coco = 2
        test_folder = '/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/test_images/'
        ## first we have to guarantee the outputs image names keep sequence consistence
        output_model_merge = []
        for idx, (a, b) in enumerate(zip(outputs[0], outputs[1])):
            print(idx)
            img_name_a = os.path.basename(a[2]['file_name'])
            img_name_b = os.path.basename(b[2]['file_name'])
            assert img_name_a == img_name_b
            img_name = os.path.join(test_folder, img_name_a)
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                bboxes_a, segms_a, six_dof_a = a[0], a[1], a[2]
                bboxes_b, segms_b, six_dof_b = b[0], b[1], b[2]
                bboxes_merge = bboxes_a.copy()
                segms_merge = segms_a.copy()
                six_dof_merge = six_dof_a.copy()

                bboxes_a_with_IOU = get_IOU(image, bboxes_a[car_cls_coco], segms_a[car_cls_coco], six_dof_a,
                                            car_id2name, self.car_model_dict, self.unique_car_mode, self.camera_matrix)
                bboxes_b_with_IOU = get_IOU(image, bboxes_b[car_cls_coco], segms_b[car_cls_coco], six_dof_b,
                                            car_id2name, self.car_model_dict, self.unique_car_mode, self.camera_matrix)
                bboxes_with_IOU = np.concatenate([bboxes_a_with_IOU, bboxes_b_with_IOU], axis=0)
                inds = nms_with_IOU(bboxes_with_IOU)  ## IOU nms filter out processing return output indices
                inds = np.array(inds)
                inds_a = inds[np.where(inds < bboxes_a_with_IOU.shape[0])]
                inds_b = inds[np.where(inds >= bboxes_a_with_IOU.shape[0])] - bboxes_a_with_IOU.shape[0]
                bboxes_merge[car_cls_coco] = np.concatenate(
                    [bboxes_a[car_cls_coco][inds_a], bboxes_b[car_cls_coco][inds_b]], axis=0)
                segms_merge[car_cls_coco] = np.concatenate(
                    [np.array(segms_a[car_cls_coco])[inds_a], np.array(segms_b[car_cls_coco])[inds_b]], axis=0)
                six_dof_merge['car_cls_score_pred'] = np.concatenate(
                    [six_dof_a['car_cls_score_pred'][inds_a], six_dof_b['car_cls_score_pred'][inds_b]], axis=0)
                six_dof_merge['quaternion_pred'] = np.concatenate(
                    [six_dof_a['quaternion_pred'][inds_a], six_dof_b['quaternion_pred'][inds_b]], axis=0)
                six_dof_merge['trans_pred_world'] = np.concatenate(
                    [six_dof_a['trans_pred_world'][inds_a], six_dof_b['trans_pred_world'][inds_b]], axis=0)
                output_model_merge.append((bboxes_merge, segms_merge, six_dof_merge))

                car_cls_score_pred = six_dof_merge['car_cls_score_pred']
                quaternion_pred = six_dof_merge['quaternion_pred']
                trans_pred_world = six_dof_merge['trans_pred_world'].copy()
                euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
                car_labels = np.argmax(car_cls_score_pred, axis=1)
                kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
                car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])
                # img_box_mesh_refined = self.visualise_box_mesh(image,bboxes[car_cls_coco], segms[car_cls_coco],car_names, euler_angle,trans_pred_world_refined)
                img_box_mesh_refined, iou_flag = self.visualise_box_mesh(image, bboxes_merge[car_cls_coco],
                                                                         segms_merge[car_cls_coco], car_names,
                                                                         euler_angle, trans_pred_world)
                imwrite(img_box_mesh_refined,
                        os.path.join(args.out[:-4] + '_mes_box_vis_merged/' + img_name.split('/')[-1])[
                        :-4] + '_merged.jpg')

        return output_model_merge

    def visualise_box_mesh(self, image, bboxes, segms, car_names, euler_angle, trans_pred_world):
        im_combime, iou_flag = draw_box_mesh_kaggle_pku(image,
                                                        bboxes,
                                                        segms,
                                                        car_names,
                                                        self.car_model_dict,
                                                        self.camera_matrix,
                                                        trans_pred_world,
                                                        euler_angle)

        return im_combime, iou_flag

    def visualise_mesh(self, image, bboxes, segms, car_names, euler_angle, trans_pred_world):

        im_combime = draw_result_kaggle_pku(image,
                                            bboxes,
                                            segms,
                                            car_names,
                                            self.car_model_dict,
                                            self.camera_matrix,
                                            trans_pred_world,
                                            euler_angle)

        return im_combime

    def visualise_kaggle(self, img, coords):
        # You will also need functions from the previous cells
        x_l = 1.02
        y_l = 0.80
        z_l = 2.31

        img = img.copy()
        for point in coords:
            # Get values
            x, y, z = point[3], point[4], point[5]
            # yaw, pitch, roll = -pitch, -yaw, -roll
            yaw, pitch, roll = point[0], point[1], point[2]
            yaw, pitch, roll = -pitch, -yaw, -roll
            # Math
            Rt = np.eye(4)
            t = np.array([x, y, z])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.array([[x_l, -y_l, -z_l, 1],
                          [x_l, -y_l, z_l, 1],
                          [-x_l, -y_l, z_l, 1],
                          [-x_l, -y_l, -z_l, 1],
                          [0, 0, 0, 1]]).T
            img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]
            img_cor_points = img_cor_points.astype(int)
            # Drawing
            img = draw_line(img, img_cor_points)
            img = draw_points(img, img_cor_points[-1:])

        return img

    def clean_corrupted_images(self, annotations):
        # For training images, there are 5 corrupted images:
        corrupted_images = ['ID_1a5a10365', 'ID_4d238ae90', 'ID_408f58e9f', 'ID_bb1d991f6', 'ID_c44983aeb']
        annotations_clean = [ann for ann in annotations if ann['filename'].split('/')[-1][:-4] not in corrupted_images]
        return annotations_clean

    def clean_outliers(self, annotations):
        """
        We get rid of the outliers in this dataset
        :
        if translation[0] < -80 or translation[0] > 80
        or translation[1] < 1 or translation[1] > 50 or
        translation[2] < 3 or translation[2] > 150

        :param train:
        :return:
        """

        corrupted_count = 0
        clean_count = 0
        annotations_clean = []

        for idx in range(len(annotations)):
            ann = annotations[idx]

            bboxes = []
            labels = []
            eular_angles = []
            quaternion_semispheres = []
            translations = []
            rles = []

            for box_idx in range(len(ann['bboxes'])):
                translation = ann['translations'][box_idx]
                if translation[0] < -80 or translation[0] > 80 or \
                        translation[1] < 1 or translation[1] > 50 \
                        or translation[2] < 3 or translation[2] > 150:
                    corrupted_count += 1
                    continue
                else:
                    bboxes.append(ann['bboxes'][box_idx])
                    labels.append(ann['labels'][box_idx])
                    eular_angles.append(ann['eular_angles'][box_idx])
                    quaternion_semispheres.append(ann['quaternion_semispheres'][box_idx])
                    translations.append(ann['translations'][box_idx])
                    rles.append(ann['rles'][box_idx])

            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            eular_angles = np.array(eular_angles, dtype=np.float32)
            quaternion_semispheres = np.array(quaternion_semispheres, dtype=np.float32)
            translations = np.array(translations, dtype=np.float32)
            assert len(bboxes) == len(labels) == len(eular_angles) == len(quaternion_semispheres) == len(translations)
            clean_count += len(bboxes)
            annotation = {
                'filename': ann['filename'],
                'width': ann['width'],
                'height': ann['height'],
                'bboxes': bboxes,
                'labels': labels,
                'eular_angles': eular_angles,
                'quaternion_semispheres': quaternion_semispheres,
                'translations': translations,
                'rles': rles
            }
            annotations_clean.append(annotation)
        print("Totaly corrupted count is: %d, clean count: %d" % (corrupted_count, clean_count))
        return annotations_clean

    def group_rectangles(self, annotations,
                         outfile='/data/Kaggle/bboxes_with_translation_pick.pkl',
                         draw_flag=True):
        """
        This will generate the referenced bboxes for translation regression. Only done onces
        :param annotations:
        :param outfile:
        :param draw_flag:
        :return:
        """

        bboxes_with_translation = []
        for idx in range(len(annotations)):
            ann = annotations[idx]
            bboxes_with_translation.append(np.concatenate((ann['bboxes'], ann['translations']), axis=1))

        bboxes_with_translation = np.vstack(bboxes_with_translation)
        print('Total number of cars: %d.' % bboxes_with_translation.shape[0])
        # We read an image first
        bboxes_with_translation_pick = non_max_suppression_fast(bboxes_with_translation, overlapThresh=0.99)
        # Some boxes are outside the boundary, we need to get rid of them:
        idx_valid = np.array(bboxes_with_translation_pick[:, 0] <= self.image_shape[1]) & \
                    np.array(bboxes_with_translation_pick[:, 1] <= self.image_shape[0]) & \
                    np.array(bboxes_with_translation_pick[:, 0] >= 0) & np.array(
            bboxes_with_translation_pick[:, 1] >= 1480)

        bboxes_with_translation_pick = bboxes_with_translation_pick[idx_valid]
        print('Final number of selected boxed: %d.' % bboxes_with_translation_pick.shape[0])
        mmcv.dump(bboxes_with_translation_pick, outfile)

        if draw_flag:
            img = imread(annotations[0]['filename'])
            img_2 = img.copy()
            for bb in bboxes_with_translation:
                img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 255, 0), thickness=1)
            imwrite(img, '/data/Kaggle/wudi_data/rect_all.jpg')
            for bb in bboxes_with_translation_pick:
                img_2 = cv2.rectangle(img_2, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 255, 0), thickness=1)
            imwrite(img_2, '/data/Kaggle/wudi_data/rect_selected.jpg')

    def print_statistics_annotations(self, annotations):
        """
        Print some statistics from annotations
        :param annotations:
        :return:
        """
        car_per_image = []
        xp, yp = [], []
        xw, yw, zw = [], [], []
        car_models = []
        for idx in range(len(annotations)):
            ann = annotations[idx]
            car_per_image.append(len(ann['bboxes']))
            for box_idx in range(len(ann['bboxes'])):
                car_models.append(ann['labels'][box_idx])
                translation = ann['translations'][box_idx]

                xpt, ypt, xwt, ywt, zwt = self._get_img_coords(translation=translation)
                xp.append(xpt)
                yp.append(ypt)
                xw.append(xwt)
                yw.append(ywt)
                zw.append(zwt)

        car_per_image = np.array(car_per_image)
        print('Total images: %d, car num sum: %d, minmin: %d, max: %d, mean: %d' %
              (len(annotations), car_per_image.sum(), car_per_image.min(), car_per_image.max(), car_per_image.mean()))
        """
        Total images: 6691, car num sum: 74029, minmin: 1, max: 43, mean: 11
        """
        xp, yp = np.array(xp), np.array(yp)
        print("x min: %d, max: %d, mean: %d" % (int(min(xp)), int(max(xp)), int(xp.mean())))
        print("y min: %d, max: %d, mean: %d" % (int(min(yp)), int(max(yp)), int(yp.mean())))
        """
        x min: -851, max: 4116, mean: 1551
        y min: 1482, max: 3427, mean: 1820
        """

        xw, yw, zw = np.array(xw), np.array(yw), np.array(zw)
        print("x min: %d, max: %d, mean: %d, std: %.3f" % (int(min(xw)), int(max(xw)), int(xw.mean()), xw.std()))
        print("y min: %d, max: %d, mean: %d, std: %.3f" % (int(min(yw)), int(max(yw)), int(yw.mean()), yw.std()))
        print("z min: %d, max: %d, mean: %d, std: %.3f" % (int(min(zw)), int(max(zw)), int(zw.mean()), zw.std()))

        """
        x min: -90, max: 519, mean: -3, std: 14.560
        y min: 1, max: 689, mean: 9, std: 6.826
        z min: 3, max: 3502, mean: 52, std: 40.046
        """

        car_models = np.array(car_models)
        print("Car model: max: %d, min: %d, total: %d" % (car_models.max(), car_models.min(), len(car_models)))
        # Car model: max: 76, min: 2, total: 49684
        print('Unique car models:')
        print(np.unique(car_models))
        # array([2, 6, 7, 8, 9, 12, 14, 16, 18, 19, 20, 23, 25, 27, 28, 31, 32,
        #        35, 37, 40, 43, 46, 47, 48, 50, 51, 54, 56, 60, 61, 66, 70, 71, 76])
        print("Number of unique car models: %d" % len(np.unique(car_models)))
        # 34

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
        
        # Clean
        x min: -79, max: 79, mean: -3, std: 14.015
        y min: 1, max: 42, mean: 9, std: 4.695
        z min: 3, max: 150, mean: 50, std: 29.596
        """
        # Next we filter our 99.9% data distribution
        xmin, xmax = -80, 80
        ymin, ymax = 1, 50
        xs_cdf = sum((xs > xmin) * (xs < xmax))
        ys_cdf = sum((ys > ymin) * (ys < ymax))
        xs_ys_cdf = sum((xs > xmin) * (xs < xmax) * (ys > ymin) * (ys < ymax))
        print('X within range (%d, %d) will have cdf of: %.6f, outlier number: %d' % (
            xmin, xmax, xs_cdf / len(xs), len(xs) - xs_cdf))
        print('Y within range (%d, %d) will have cdf of: %.6f, outlier number: %d' % (
            ymin, ymax, ys_cdf / len(ys), len(ys) - ys_cdf))

        print('Both will have cdf of: %.6f, outlier number: %d' % (xs_ys_cdf / len(ys), len(ys) - xs_ys_cdf))

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

    def _get_img_coords(self, s=None, translation=None):
        '''
        Input is a PredictionString (e.g. from train dataframe)
        Output is two arrays:
            xs: x coordinates in the image
            ys: y coordinates in the image
        '''
        if translation is not None:
            xs, ys, zs = translation
            P = np.array([xs, ys, zs]).T
            img_p = np.dot(self.camera_matrix, P).T
            img_p[0] /= img_p[2]
            img_p[1] /= img_p[2]
            return img_p[0], img_p[1], xs, ys, zs

        else:
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
        gt_class_labels = []  # this will always be fixed as car class
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        eular_angles = []
        quaternion_semispheres = []
        translations = []
        if self.rotation_augmenation:
            # We follow the camera rotation augmentation as in
            # https://www.kaggle.com/outrunner/rotation-augmentation
            # It's put in here (instead of dataset.pipeline.transform is because
            # we need car model json
            alpha = ((np.random.random() ** 0.5) * 8 - 5.65) * np.pi / 180.
            beta = (np.random.random() * 50 - 25) * np.pi / 180.
            gamma = (np.random.random() * 6 - 3) * np.pi / 180. + beta / 3

            Mat, Rot = self.rotateImage(alpha, beta, gamma)
        else:
            Mat, Rot = self.rotateImage(0, 0, 0)

        for i in range(len(ann_info['bboxes'])):
            x1, y1, x2, y2 = ann_info['bboxes'][i]
            w, h = x2 - x1, y2 - y1
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
            if ann_info.get('iscrowd', False):  # TODO: train mask need to include
                gt_bboxes_ignore.append(bbox)
            else:
                if not self.rotation_augmenation:

                    gt_bboxes.append(bbox)
                    gt_label = self.cat2label[ann_info['labels'][i]]
                    gt_labels.append(gt_label)
                    gt_class_labels.append(3)  # coco 3 is "car" class
                    mask = maskUtils.decode(ann_info['rles'][i])
                    gt_masks_ann.append(mask)

                    eular_angles.append(ann_info['eular_angles'][i])
                    quaternion_semispheres.append(ann_info['quaternion_semispheres'][i])
                    translations.append(translation)
                else:

                    yaw, pitch, roll = ann_info['eular_angles'][i]
                    r1 = R.from_euler('xyz', [-pitch, -yaw, -roll], degrees=False)
                    r2 = R.from_euler('xyz', [beta, -alpha, -gamma], degrees=False)

                    pitch_rot, yaw_rot, roll_rot = (r2 * r1).as_euler('xyz') * (-1)
                    eular_angle_rot = np.array([yaw_rot, pitch_rot, roll_rot])
                    quaternion_rot = euler_angles_to_quaternions(eular_angle_rot)
                    quaternion_semisphere_rot = quaternion_upper_hemispher(quaternion_rot)
                    quaternion_semisphere_rot = np.array(quaternion_semisphere_rot, dtype=np.float32)

                    x, y, z = translation
                    x_rot, y_rot, z_rot, _ = np.dot(Rot, [x, y, z, 1])
                    translation_rot = np.array([x_rot, y_rot, z_rot])

                    car_name = car_id2name[ann_info['labels'][i]].name
                    vertices = np.array(self.car_model_dict[car_name]['vertices'])
                    vertices[:, 1] = -vertices[:, 1]
                    triangles = np.array(self.car_model_dict[car_name]['faces']) - 1

                    bbox_rot, mask_rot = self.get_box_and_mask(eular_angle_rot, translation_rot, vertices, triangles)
                    # Some rotated bbox might be out of the image
                    if bbox_rot[2] < 0 or bbox_rot[3] < 0 or \
                            bbox_rot[0] > self.image_shape[0] or bbox_rot[1] > self.image_shape[1] - self.bottom_half:
                        continue

                    gt_label = self.cat2label[ann_info['labels'][i]]

                    gt_bboxes.append(bbox_rot)
                    gt_labels.append(gt_label)
                    gt_class_labels.append(3)  # coco 3 is "car" class
                    gt_masks_ann.append(mask_rot)

                    eular_angles.append(eular_angle_rot)
                    quaternion_semispheres.append(quaternion_semisphere_rot)
                    translations.append(translation_rot)

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

            eular_angles=eular_angles,
            quaternion_semispheres=quaternion_semispheres,
            translations=translations,
            Mat=Mat)

        return ann

    def rotateImage(self, alpha=0, beta=0, gamma=0):

        fx, dx = self.camera_matrix[0, 0], self.camera_matrix[0, 2]
        fy, dy = self.camera_matrix[1, 1], self.camera_matrix[1, 2]

        # Projection 2D -> 3D matrix
        A1 = np.array([[1 / fx, 0, -dx / fx],
                       [0, 1 / fx, -dy / fx],
                       [0, 0, 1],
                       [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0],
                       [0, cos(alpha), -sin(alpha), 0],
                       [0, sin(alpha), cos(alpha), 0],
                       [0, 0, 0, 1]])

        RY = np.array([[cos(beta), 0, -sin(beta), 0],
                       [0, 1, 0, 0],
                       [sin(beta), 0, cos(beta), 0],
                       [0, 0, 0, 1]])
        RZ = np.array([[cos(gamma), -sin(gamma), 0, 0],
                       [sin(gamma), cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(RZ, np.dot(RX, RY))

        # 3D -> 2D matrix
        A2 = np.array([[fx, 0, dx, 0],
                       [0, fy, dy, 0],
                       [0, 0, 1, 0]])
        # Final transformation matrix
        trans = np.dot(A2, np.dot(R, A1))
        # Apply matrix transformation
        return trans, R

    def get_box_and_mask(self, eular_angle, translation, vertices, triangles):
        # project 3D points to 2d image plane
        yaw, pitch, roll = eular_angle
        # I think the pitch and yaw should be exchanged
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = translation
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
        x1, y1, x2, y2 = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), \
                         img_cor_points[:, 0].max(), img_cor_points[:, 1].max()
        bbox = np.array([x1, y1, x2, y2])
        if self.bottom_half:
            # we only take bottom half image
            bbox = [x1, y1 - self.bottom_half, x2, y2 - self.bottom_half]
        #### Now draw the mask
        # project 3D points to 2d image plane
        mask_seg = np.zeros(self.image_shape, dtype=np.uint8)
        mask_seg_mesh = np.zeros(self.image_shape, dtype=np.uint8)
        for t in triangles:
            coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                             dtype=np.int32)
            # This will draw the mask for segmenation
            cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
            cv2.polylines(mask_seg_mesh, np.int32([coord]), 1, (0, 255, 0))

        ground_truth_binary_mask = np.zeros(mask_seg.shape, dtype=np.uint8)
        ground_truth_binary_mask[mask_seg == 255] = 1
        if self.bottom_half > 0:  # this indicate w
            ground_truth_binary_mask = ground_truth_binary_mask[int(self.bottom_half):, :]

        # kernel_size = int(((y2 - y1) / 2 + (x2 - x1) / 2) / 10)
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # # Following is the code to find mask
        # ground_truth_binary_mask = cv2.dilate(ground_truth_binary_mask, kernel, iterations=1)
        # ground_truth_binary_mask = cv2.erode(ground_truth_binary_mask, kernel, iterations=1)
        return bbox, ground_truth_binary_mask

    def visualise_pred(self, outputs, args):
        car_cls_coco = 2

        for idx in tqdm(range(len(self.annotations))):
            ann = self.annotations[idx]
            img_name = ann['filename']
            if not os.path.isfile(img_name):
                assert "Image file does not exist!"
            else:
                image = imread(img_name)
                output = outputs[idx]
                # output is a tuple of three elements
                bboxes, segms, six_dof = output[0], output[1], output[2]
                car_cls_score_pred = six_dof['car_cls_score_pred']
                quaternion_pred = six_dof['quaternion_pred']
                trans_pred_world = six_dof['trans_pred_world']
                euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
                car_labels = np.argmax(car_cls_score_pred, axis=1)
                kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
                car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

                assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                       == len(trans_pred_world) == len(euler_angle) == len(car_names)
                # now we start to plot the image from kaggle
                im_combime, iou_flag = self.visualise_box_mesh(image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                               car_names, euler_angle, trans_pred_world)
                imwrite(im_combime, os.path.join(args.out[:-4] + '_mes_vis/' + img_name.split('/')[-1]))

    def visualise_pred_single_node(self, idx, outputs, args):
        car_cls_coco = 2

        ann = self.annotations[idx]
        img_name = ann['filename']
        if not os.path.isfile(img_name):
            assert "Image file does not exist!"
        else:
            image = imread(img_name)
            output = outputs[idx]
            # output is a tuple of three elements
            bboxes, segms, six_dof = output[0], output[1], output[2]
            car_cls_score_pred = six_dof['car_cls_score_pred']
            quaternion_pred = six_dof['quaternion_pred']
            trans_pred_world = six_dof['trans_pred_world']
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
            car_labels = np.argmax(car_cls_score_pred, axis=1)
            kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
            car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

            assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
                   == len(trans_pred_world) == len(euler_angle) == len(car_names)
            # now we start to plot the image from kaggle
            im_combime, iou_flag = self.visualise_box_mesh(image, bboxes[car_cls_coco], segms[car_cls_coco], car_names,
                                                           euler_angle, trans_pred_world)
            imwrite(im_combime, os.path.join(args.out[:-4] + '_mes_vis/' + img_name.split('/')[-1]))

    def pkl_postprocessing_restore_xyz_multiprocessing(self, outputs):
        """
        Post processing of storing x,y using z prediction (YYJ method)
        We use multiprocessing thread here
        Args:
            outputs: pkl file generated from a single model

        Returns:

        """
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)

        outputs_refined = []

        for output_refined in pool.imap(self.restore_pool, [(idx, output) for idx, output in enumerate(outputs)]):
            # print('output_refined',output_refined)
            outputs_refined.append(output_refined)

        return outputs_refined

    def restore_pool(self, t):
        return self.restore_xyz_withIOU_single(*t)

    def restore_xyz_withIOU_single(self, idx, output_origin, car_cls_coco=2):
        output = copy.deepcopy(output_origin)
        print('idx', idx)
        bboxes, segms, six_dof = output[0], output[1], output[2]
        car_cls_score_pred = six_dof['car_cls_score_pred']
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world'].copy()
        euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
        car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])

        assert len(bboxes[car_cls_coco]) == len(segms[car_cls_coco]) == len(kaggle_car_labels) \
               == len(trans_pred_world) == len(euler_angle) == len(car_names)

        quaternion_semisphere_refined, flag = self.refine_yaw_and_roll(bboxes[car_cls_coco],
                                                                       euler_angle,
                                                                       quaternion_pred)
        if flag:
            output[2]['quaternion_pred'] = quaternion_semisphere_refined
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])

        trans_pred_world_refined = self.restore_x_y_from_z_withIOU(bboxes[car_cls_coco],
                                                                   segms[car_cls_coco],
                                                                   car_names,
                                                                   euler_angle,
                                                                   trans_pred_world,
                                                                   self.car_model_dict,
                                                                   self.camera_matrix)

        # print('change ',trans_pred_world,trans_pred_world_refined)
        output[2]['trans_pred_world'] = trans_pred_world_refined

        return output

    def refine_yaw_and_roll(self,
                            bboxes,
                            euler_angle,
                            quaternion_pred,
                            score_thr=0.1,
                            roll_threshold=0.2,
                            yaw_threshold=(0, 0.3)):
        """
        we find that sometimes the predicted roll or yaw is out of normal range,so we confine it to normal range.
        roll mainly locates from -0.1 to 0.1 we confine the value out of absolute value of 0.2
        Args:
            bboxes:
            segms:
            class_names:
            euler_angle:
            quaternion_pred:
            trans_pred_world:
            car_model_dict:
            camera_matrix:
            score_thr:
            roll_threshold:
            yaw_threshold:

        Returns:

        """

        pi = np.pi
        flag = False
        candidates = np.array([-pi, 0, pi])
        euler_angle_refined = euler_angle.copy()
        quaternion_pred_refined = quaternion_pred.copy()
        for bbox_idx in range(len(bboxes)):
            if bboxes[bbox_idx, -1] <= score_thr:  ## we only restore case when score > score_thr(0.1)
                continue
            ea = euler_angle_refined[bbox_idx]
            yaw, pitch, roll = ea
            candidate_roll = candidates[np.argmin(np.abs(candidates - roll))]
            if yaw < yaw_threshold[0] or yaw > yaw_threshold[1] or np.abs(roll - candidate_roll) > roll_threshold:
                if yaw < yaw_threshold[0]:
                    # print('yaw change',yaw,yaw_threshold[0])
                    yaw = 0.15  # waited to be determined

                if yaw > yaw_threshold[1]:
                    # print('yaw change',yaw,yaw_threshold[1])
                    yaw = 0.15  # waited to be determined

                if np.abs(roll - candidate_roll) > roll_threshold:
                    # print('roll',roll,candidate_roll)

                    roll = candidate_roll

                quaternion_refined = euler_angles_to_quaternions(np.array([yaw, pitch, roll]))
                quaternion_semisphere_refined = quaternion_upper_hemispher(quaternion_refined)
                quaternion_pred_refined[bbox_idx] = np.array(quaternion_semisphere_refined)
                flag = True

        return quaternion_pred_refined, flag

    def restore_x_y_from_z_withIOU(self,
                                   bboxes,
                                   segms,
                                   class_names,
                                   euler_angle,
                                   trans_pred_world,
                                   car_model_dict,
                                   camera_matrix,
                                   score_thr=0.1,
                                   refined_threshold1=10,
                                   refined_threshold2=28,
                                   IOU_threshold=0.3):

        image_shape = np.array(self.image_shape)  # this is generally the case
        if self.bottom_half:
            image_shape[0] -= self.bottom_half
        trans_pred_world_refined = trans_pred_world.copy()
        for bbox_idx in range(len(bboxes)):
            if bboxes[bbox_idx, -1] <= score_thr:  ## we only restore case when score > score_thr(0.1)
                continue

            bbox = bboxes[bbox_idx]
            ## below is the predicted mask
            mask_all_pred = np.zeros(image_shape)  ## this is the background mask
            mask_all_mesh = np.zeros(image_shape)
            mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
            mask_all_pred += mask_pred
            mask_all_pred_area = np.sum(mask_all_pred == 1)

            t = trans_pred_world[bbox_idx]
            t_refined = self.get_xy_from_z(bbox, t)

            score_iou_mask_before, score_iou_before = self.get_iou_score(bbox_idx, car_model_dict, camera_matrix,
                                                                         class_names,
                                                                         mask_all_pred, mask_all_mesh,
                                                                         mask_all_pred_area,
                                                                         euler_angle, t)
            score_iou_mask_after, score_iou_after = self.get_iou_score(bbox_idx, car_model_dict, camera_matrix,
                                                                       class_names,
                                                                       mask_all_pred, mask_all_mesh, mask_all_pred_area,
                                                                       euler_angle, t_refined)
            if t[2] > refined_threshold2:
                trans_pred_world_refined[bbox_idx] = t_refined
            elif t[2] < refined_threshold1:
                if score_iou_before < IOU_threshold:
                    print('score_iou_before', score_iou_before)
                    continue
            else:
                if score_iou_after - score_iou_before > 0.05:
                    print('score good', score_iou_before, score_iou_after)
                    trans_pred_world_refined[bbox_idx] = t_refined
            if score_iou_after < IOU_threshold:
                ## we filter out candidate with IOU <=0.3
                print('score_iou_after', score_iou_after)
                continue

        return trans_pred_world_refined

    def get_xy_from_z(self, boxes, t):
        boxes_copy = boxes.copy()
        x, y, z = t
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        boxes_copy[1::2] += self.bottom_half
        center = np.array([np.mean(boxes_copy[:-1][0::2]), np.mean(boxes_copy[1::2])])
        X = (center[0] - cx) * z / fx
        Y = (center[1] - cy) * z / fy

        return np.array([X, Y, z])

    def get_iou_score(self, bbox_idx, car_model_dict, camera_matrix, class_names,
                      mask_all_pred, mask_all_mesh, mask_all_pred_area, euler_angle, t):
        vertices = np.array(car_model_dict[class_names[bbox_idx]]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(car_model_dict[class_names[bbox_idx]]['faces']) - 1

        ea = euler_angle[bbox_idx]
        yaw, pitch, roll = ea[0], ea[1], ea[2]
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        mask_all_mesh_tmp = mask_all_mesh.copy()
        for tri in triangles:
            coord = np.array([img_cor_points[tri[0]][:2], img_cor_points[tri[1]][:2], img_cor_points[tri[2]][:2]],
                             dtype=np.int32)
            coord[:, 1] -= 1480
            cv2.drawContours(mask_all_mesh_tmp, np.int32([coord]), 0, 1, -1)
            # cv2.drawContours(img,np.int32([coord]),0,color,-1)

        intersection_area = np.sum(mask_all_pred * mask_all_mesh_tmp)
        union_area = np.sum(np.logical_or(mask_all_pred, mask_all_mesh_tmp))
        iou_mask_score = intersection_area / mask_all_pred_area
        iou_score = intersection_area / union_area
        return iou_mask_score, iou_score

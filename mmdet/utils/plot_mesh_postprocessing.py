import os
import json

import cv2
import copy
import mmcv
from mmcv.image import imread, imwrite
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch
from torchvision import transforms


from mmdet.datasets.car_models import car_id2name
from mmdet.datasets.kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, quaternion_upper_hemispher, \
    quaternion_to_euler_angle
from demo.visualisation_utils import draw_box_mesh_kaggle_pku, refine_yaw_and_roll, restore_x_y_from_z_withIOU
from AAE.aae import AugmentedAutoEncoder
from AAE.knn import KNNClassifier


class Plot_Mesh_Postprocessing:
    def __init__(self,
                 car_model_json_dir='/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data',
                 test_image_folder='/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/test_images/'):
        """
        YYJ post processing script --> using Z to modify x, y
        Args:
            car_model_json_dir: car json file directory
            test_image_folder:  test image folder
        """
        # some hard coded parameters
        self.car_model_json_dir = car_model_json_dir
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
        self.test_image_folder = test_image_folder

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

    def load_car_models(self):
        car_model_dir = os.path.join(self.car_model_json_dir, 'car_models_json')
        car_model_dict = {}
        for car_name in tqdm(os.listdir(car_model_dir)):
            with open(os.path.join(self.car_model_json_dir, 'car_models_json', car_name)) as json_file:
                car_model_dict[car_name[:-5]] = json.load(json_file)

        return car_model_dict

    def restore_pool(self, t):
        # print('t',t)
        return self.restore_xyz_withIOU_single(*t)

    def visualise_pred_postprocessing_multiprocessing(self, outputs):
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)

        outputs_refined = []

        for output_refined in pool.imap(self.restore_pool, [(idx, output) for idx, output in enumerate(outputs)]):
            # print('output_refined',output_refined)
            outputs_refined.append(output_refined)

        return outputs_refined

    def restore_xyz_withIOU_single(self, idx, output_origin, car_cls_coco=2):
        output = copy.deepcopy(output_origin)
        print('idx', idx)
        img_name = os.path.join(self.test_image_folder, os.path.basename(output[2]['file_name']))
        image = imread(img_name)
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
        # now we start to plot the image from kaggle
        quaternion_semisphere_refined, flag = refine_yaw_and_roll(image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                                  car_names, euler_angle, quaternion_pred,
                                                                  trans_pred_world,
                                                                  self.car_model_dict,
                                                                  self.camera_matrix)
        if flag:
            output[2]['quaternion_pred'] = quaternion_semisphere_refined
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])

        trans_pred_world_refined = restore_x_y_from_z_withIOU(image, bboxes[car_cls_coco], segms[car_cls_coco],
                                                              car_names, euler_angle, trans_pred_world,
                                                              self.car_model_dict,
                                                              self.camera_matrix)

        # print('change ',trans_pred_world,trans_pred_world_refined)
        output[2]['trans_pred_world'] = trans_pred_world_refined

        return output

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

    def load_anno_idx(self, idx, img_concat, train,
                      draw_dir='/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/crop_visualization/crop_mesh'):

        bboxes = []
        img1, img2, img3 = img_concat
        mask_all = np.zeros(img1.shape)
        merged_image1 = img1.copy()
        merged_image2 = img2.copy()
        merged_image3 = img3.copy()
        alpha = 0.8  # transparency

        gt = self._str2coords(train['PredictionString'].iloc[idx])
        for gt_pred in gt:
            eular_angle = np.array([gt_pred['yaw'], gt_pred['pitch'], gt_pred['roll']])

            translation = np.array([gt_pred['x'], gt_pred['y'], gt_pred['z']])
            quaternion = euler_angles_to_quaternions(eular_angle)
            quaternion_semisphere = quaternion_upper_hemispher(quaternion)

            new_eular_angle = quaternion_to_euler_angle(quaternion_semisphere)

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
            x1, y1, x2, y2 = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(), img_cor_points[:,
                                                                                     0].max(), img_cor_points[:,
                                                                                               1].max()
            bboxes.append([x1, y1, x2, y2])

            # project 3D points to 2d image plane
            mask_seg = np.zeros(img1.shape, dtype=np.uint8)
            mask_seg_mesh = np.zeros(img1.shape, dtype=np.uint8)
            for t in triangles:
                coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                                 dtype=np.int32)
                # This will draw the mask for segmenation
                cv2.drawContours(mask_seg, np.int32([coord]), 0, (0, 0, 255), -1)
                # cv2.polylines(mask_seg_mesh, np.int32([coord]), 1, (0, 255, 0))

            mask_all += mask_seg

        # if False:
        mask_all = mask_all * 255 / mask_all.max()
        cv2.addWeighted(img1.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image1)
        cv2.addWeighted(img2.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image2)
        cv2.addWeighted(img3.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image3)

        imwrite(merged_image1, os.path.join(draw_dir, train['ImageId'].iloc[idx] + '_1.jpg'))
        imwrite(merged_image2, os.path.join(draw_dir, train['ImageId'].iloc[idx] + '_2.jpg'))
        imwrite(merged_image3, os.path.join(draw_dir, train['ImageId'].iloc[idx] + '_3.jpg'))


camera_matrix_xiaomi10pro = np.array([[450.63, 0, 321.47],
                                      [0, 450.60, 239.81],
                                      [0, 0, 1]], dtype=np.float32)

camera_matrix_mate30 = np.array([[493.90, 0, 318.69],
                                 [0, 493.81, 240.39],
                                 [0, 0, 1]], dtype=np.float32)

camera_matrix_xiaomiMix3 = np.array([[481.28, 0, 321.06],
                                     [0, 481.25, 239.46],
                                     [0, 0, 1]], dtype=np.float32)


class Plot_Mesh_Postprocessing_Car_Insurance:
    def __init__(self,
                 camera_matrix=None,
                 ZRENDER=0.2,
                 SCALE=1 / 20,
                 car_model_name='aodi-Q7-SUV',
                 car_model_json_dir='/data/Kaggle/pku-autonomous-driving/car_models_json'):
        """
        We follow the paper to acquire the  Projective Distance Estimation
        Eq.5 and Eq. 6 from
        "Implicit 3D Orientation Learning for 6D Object Detection from RGB Images"
        Args:
            car_model_json_dir: car json file directory
            test_image_folder:  test image folder
        """
        # some hard coded parameters
        self.car_model_json_dir = car_model_json_dir
        self.image_shape = (640, 480)  # this is generally the case

        # From camera --> this is wudi's Xiaomi 10
        self.camera_matrix = camera_matrix

        print("Loading Car model files...")
        self.car_model_name = car_model_name
        self.car_model_dict = self.load_car_models_one_car(car_model_name)
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        self.cat2label = {car_model: i for i, car_model in enumerate(self.unique_car_mode)}

        # Render configuration
        # We render the synthetic rendering at 0.2 meteres
        self.ZRENDER = ZRENDER
        self.TRANSLATION_RENDER = np.array([0, 0, self.ZRENDER])
        self.SCALE = SCALE

    def load_car_models_one_car(self, car_name):
        car_model_dict = {}
        with open(os.path.join(self.car_model_json_dir, car_name + '.json')) as json_file:
            car_model_dict[car_name] = json.load(json_file)

        return car_model_dict

    def projectiveDistanceEstimation(self, output_origin, image_path, draw=False):

        # Computing the diagonal length will take roughly 0.1 second per car
        # import time
        # start = time.time()
        car_diagonal_rendered, bb_center_syn_x, bb_center_syn_y, img_cor_points, triangles, Rt, vertices = self.get_car_diagonal(output_origin)
        # end = time.time()
        # print(end - start)

        car_diagonal_test = np.sqrt((output_origin['x2'] - output_origin['x1']) ** 2 +
                                    (output_origin['y2'] - output_origin['y1']) ** 2)

        t_pred_z = self.ZRENDER * car_diagonal_rendered / car_diagonal_test

        bb_cent_test_x = (output_origin['x2'] + output_origin['x1']) / 2
        bb_cent_test_y = (output_origin['y2'] + output_origin['y1']) / 2

        t_pred_x = t_pred_z / self.camera_matrix[0, 0] * (bb_cent_test_x - bb_center_syn_x)
        t_pred_y = t_pred_z / self.camera_matrix[1, 1] * (bb_cent_test_y - bb_center_syn_y)

        if draw:
            translation_vector = np.array([t_pred_x, t_pred_y, t_pred_z])
            self.draw_rendered_image(image_path, Rt, translation_vector, vertices, triangles)
        return t_pred_x, t_pred_y, t_pred_z

    def draw_rendered_image(self, image_path, Rt, translation_vector, vertices, triangles):
        img_origin = mmcv.imread(image_path)
        # Draw the render mesh
        color_ndarray = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        color = tuple([int(i) for i in color_ndarray[0]])

        Rt[:3, 3] = translation_vector
        Rt = Rt[:3, :]

        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        for tri in triangles:
            coord = np.array([img_cor_points[tri[0]][:2], img_cor_points[tri[1]][:2], img_cor_points[tri[2]][:2]], dtype=np.int32)
            cv2.polylines(img_origin, np.int32([coord]), 1, color, thickness=1)

        mmcv.imwrite(img_origin, image_path[:-4] + '_mesh.jpg')

    def get_car_diagonal(self, output_origin, positive_y=True):
        # Get all the vertices from the car mode
        vertices = np.array(self.car_model_dict[self.car_model_name]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        if positive_y:
            # Move the car to the positive plane
            vertices[:, 1] -= vertices[:, 1].min()
        vertices *= self.SCALE
        triangles = np.array(self.car_model_dict[self.car_model_name]['faces']) - 1
        ea = output_origin['rotation']
        yaw, pitch, roll = ea[0], ea[1], ea[2]
        yaw, pitch, roll = -pitch, -yaw, -roll

        Rt = np.eye(4)
        Rt[:3, 3] = self.TRANSLATION_RENDER
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        car_diagonal_rendered = np.sqrt((img_cor_points[:, 0].max() - img_cor_points[:, 0].min()) ** 2 +
                                        (img_cor_points[:, 1].max() - img_cor_points[:, 1].min()) ** 2)

        bb_center_syn_x = (img_cor_points[:, 0].max() + img_cor_points[:, 0].min()) / 2
        bb_center_syn_y = (img_cor_points[:, 1].max() + img_cor_points[:, 1].min()) / 2

        return car_diagonal_rendered, bb_center_syn_x, bb_center_syn_y, img_cor_points, triangles, Rt, vertices


class Plot_Mesh_Postprocessing_Car_Insurance_AAE:
    def __init__(self,
                 camera_matrix=None,
                 ZRENDER=0.2,
                 SCALE=1 / 20,
                 car_model_name='aodi-Q7-SUV',
                 car_model_json_dir='/data/Kaggle/pku-autonomous-driving/car_models_json'):
        """
        We follow the paper to acquire the  Projective Distance Estimation
        Eq.5 and Eq. 6 from
        "Implicit 3D Orientation Learning for 6D Object Detection from RGB Images"
        Args:
            car_model_json_dir: car json file directory
            test_image_folder:  test image folder
        """
        # some hard coded parameters
        self.car_model_json_dir = car_model_json_dir
        self.image_shape = (640, 480)  # this is generally the case

        # From camera --> this is wudi's Xiaomi 10
        self.camera_matrix = camera_matrix

        print("Loading Car model files...")
        self.car_model_name = car_model_name
        self.car_model_dict = self.load_car_models_one_car(car_model_name)
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        self.cat2label = {car_model: i for i, car_model in enumerate(self.unique_car_mode)}

        # Render configuration
        # We render the synthetic rendering at 0.2 meteres
        self.ZRENDER = ZRENDER
        self.TRANSLATION_RENDER = np.array([0, 0, self.ZRENDER])
        self.SCALE = SCALE

    def load_car_models_one_car(self, car_name):
        car_model_dict = {}
        with open(os.path.join(self.car_model_json_dir, car_name + '.json')) as json_file:
            car_model_dict[car_name] = json.load(json_file)

        return car_model_dict

    def projectiveDistanceEstimation(self, output_origin, image_path, draw=False):

        # We first use AAE to modified the predicted image file
        # loading data
        AAE_root = "/home/wudi/code/AAE/checkpoint"
        gt_tensor_path = os.path.join(AAE_root, "gt_implict_tensors.npy")
        gt_label_path = os.path.join(AAE_root, "gt_labels.npy")
        ckpt_path = os.path.join(AAE_root, "AAE_epoch1499_0818.pth")

        gt_tensors = np.load(gt_tensor_path)
        gt_labels = np.load(gt_label_path)

        # 1. 先用高版本进行下面操作，保存新c的.pth
        # model = torch.load(ckpt_path)
        # torch.save(model.state_dict(), ckpt_path[:-4]+"_state_dict.pth")
        # model = model.cuda()

        # 2. 再用低版本load_state_dict
        model = AugmentedAutoEncoder().cuda()
        model.load_state_dict(torch.load(ckpt_path[:-4] + "_state_dict.pth"))

        # knn classifier initialize
        knn_classifier = KNNClassifier(k=15)
        knn_classifier = knn_classifier.fit(gt_tensors, gt_labels)

        # transforms
        my_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # test images (cropped by detector)
        img = cv2.imread(image_path)
        img_car = img[int(output_origin['y1']):int(output_origin['y2']),
                  int(output_origin['x1']): int(output_origin['x2']), :]

        img_car = cv2.resize(img_car, (256, 256))
        input_tensor = my_transforms(img_car)

        # model inference
        pred = model.encoder(input_tensor.unsqueeze(dim=0).cuda())
        b, c, h, w = pred.size()
        pred = pred.view(-1, c * h * w)
        pred = model.fc1(pred)
        cpu_pred = pred.cpu()

        # knn classifier predict
        pred_azimuth = knn_classifier.predict(cpu_pred.detach().numpy())

        corrected_yaw = pred_azimuth * np.pi / 180
        output_origin['rotation'][1] = corrected_yaw
        # Computing the diagonal length will take roughly 0.1 second per car
        # import time
        # start = time.time()
        car_diagonal_rendered, bb_center_syn_x, bb_center_syn_y, img_cor_points, triangles, Rt, vertices = self.get_car_diagonal(output_origin)
        # end = time.time()
        # print(end - start)

        car_diagonal_test = np.sqrt((output_origin['x2'] - output_origin['x1']) ** 2 +
                                    (output_origin['y2'] - output_origin['y1']) ** 2)

        t_pred_z = self.ZRENDER * car_diagonal_rendered / car_diagonal_test

        bb_cent_test_x = (output_origin['x2'] + output_origin['x1']) / 2
        bb_cent_test_y = (output_origin['y2'] + output_origin['y1']) / 2

        t_pred_x = t_pred_z / self.camera_matrix[0, 0] * (bb_cent_test_x - bb_center_syn_x)
        t_pred_y = t_pred_z / self.camera_matrix[1, 1] * (bb_cent_test_y - bb_center_syn_y)

        if draw:
            translation_vector = np.array([t_pred_x, t_pred_y, t_pred_z])
            self.draw_rendered_image(image_path, Rt, translation_vector, vertices, triangles)
        return t_pred_x, t_pred_y, t_pred_z

    def draw_rendered_image(self, image_path, Rt, translation_vector, vertices, triangles):
        img_origin = mmcv.imread(image_path)
        # Draw the render mesh
        color_ndarray = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        color = tuple([int(i) for i in color_ndarray[0]])

        Rt[:3, 3] = translation_vector
        Rt = Rt[:3, :]

        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        for tri in triangles:
            coord = np.array([img_cor_points[tri[0]][:2], img_cor_points[tri[1]][:2], img_cor_points[tri[2]][:2]], dtype=np.int32)
            cv2.polylines(img_origin, np.int32([coord]), 1, color, thickness=1)

        mmcv.imwrite(img_origin, image_path[:-4] + '_mesh.jpg')

    def get_car_diagonal(self, output_origin, positive_y=True):
        # Get all the vertices from the car mode
        vertices = np.array(self.car_model_dict[self.car_model_name]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        if positive_y:
            # Move the car to the positive plane
            vertices[:, 1] -= vertices[:, 1].min()
        vertices *= self.SCALE
        triangles = np.array(self.car_model_dict[self.car_model_name]['faces']) - 1
        ea = output_origin['rotation']
        yaw, pitch, roll = ea[0], ea[1], ea[2]
        yaw, pitch, roll = -pitch, -yaw, -roll

        Rt = np.eye(4)
        Rt[:3, 3] = self.TRANSLATION_RENDER
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        car_diagonal_rendered = np.sqrt((img_cor_points[:, 0].max() - img_cor_points[:, 0].min()) ** 2 +
                                        (img_cor_points[:, 1].max() - img_cor_points[:, 1].min()) ** 2)

        bb_center_syn_x = (img_cor_points[:, 0].max() + img_cor_points[:, 0].min()) / 2
        bb_center_syn_y = (img_cor_points[:, 1].max() + img_cor_points[:, 1].min()) / 2

        return car_diagonal_rendered, bb_center_syn_x, bb_center_syn_y, img_cor_points, triangles, Rt, vertices


if __name__ == '__main__':
    test_folder = '/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/crop_visualization'
    ImageId = ['ID_00ac30455', 'ID_00cfeca4c', 'ID_00f8d4e89', 'ID_0a1eb2c76']
    train = pd.read_csv('/data/home/yyj/code/kaggle/new_code/Kaggle_PKU_Baidu/data/pku_data/train.csv')

    plot_mesh = Plot_Mesh_Postprocessing()
    for idx in range(len(train)):
        filename = train['ImageId'].iloc[idx]
        if filename not in ImageId:
            continue
        filename = os.path.join(test_folder, filename)
        img1 = cv2.imread(filename + '.jpg')
        img2 = cv2.imread(filename + '_1.jpg')
        img3 = cv2.imread(filename + '_2.jpg')
        img_concat = [img1, img2, img3]
        plot_mesh.load_anno_idx(idx, img_concat, train)

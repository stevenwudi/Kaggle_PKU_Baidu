import os
import numpy as np
import json
from tqdm import tqdm

import mmcv


from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle

CAR_IDX = 2
unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                   19, 20, 23, 25, 27, 28, 31, 32,
                   35, 37, 40, 43, 46, 47, 48, 50,
                   51, 54, 56, 60, 61, 66, 70, 71, 76]


def write_pose_to_json(out_pkl, output_dir, thresh=0.1, ignored_mask_binary=None, iou_ignore_threshold=None):
    """
    Args:
        im_name:
        output_dir:
        thresh:
        ignored_mask_binary:
        iou_ignore_threshold:

    Returns:

    """
    outputs = mmcv.load(out_pkl)

    for output in tqdm(outputs):
        # First we collect all the car instances info. in an image
        bboxes, segms, six_dof = output[0], output[1], output[2]
        boxes = bboxes[CAR_IDX]
        im_name = six_dof['file_name'].split('/')[-1][:-4]
        json_file = os.path.join(output_dir, im_name + '.json')
        # if os.path.exists(json_file):
        #     continue
        car_cls_score_pred = six_dof['car_cls_score_pred']
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world']
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        euler_angles = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
        car_list = []
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, -1]) < thresh:
            with open(json_file, 'w') as outfile:
                json.dump(car_list, outfile, indent=4)
        # From largest to smallest order
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

        for i in sorted_inds:
            score = boxes[i, -1]
            if score < thresh:
                continue

            # car_cls
            euler_angle_i = euler_angles[i]
            trans_pred_i = trans_pred_world[i]
            car_model_i = unique_car_mode[car_labels[i]]

            # filter out by ignored_mask_binary
            car_info = dict()
            car_info["car_id"] = int(car_model_i)
            car_info["pose"] = [float(x) for x in euler_angle_i] + [float(x) for x in trans_pred_i]
            # We use rectangle area
            car_info["area"] = int(areas[i])
            car_info["score"] = float(score)
            if iou_ignore_threshold:
                masks = np.zeros_like(ignored_mask_binary)
                masks[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]): int(boxes[i][2])] = 1
                iou_mask = masks * ignored_mask_binary
                iou = np.sum(iou_mask) / int(areas[i])
                if iou <= iou_ignore_threshold:
                    car_list.append(car_info)
                else:
                    print('This mask has been ignored')
            else:
                car_list.append(car_info)

        with open(json_file, 'w') as outfile:
            json.dump(car_list, outfile, indent=4)

    return True


if __name__ == '__main__':
    #out_pkl ='/data/Kaggle/wudi_data/ApolloScape_1041_Jan18-19-45_epoch_136.pkl'
    out_pkl = '/data/Kaggle/wudi_data/ApolloScapes/imagesall_cwxe99_3070100flip05resumme93Dec29-16-28-48_epoch_100_valid_200.pkl'

    #out_pkl = '/data/Kaggle/wudi_data/ApolloScapes/imagesall_cwxe99_3070100flip05resumme93Dec29-16-28-48_epoch_100_valid_200_refined.pkl'
    output_dir = out_pkl[:-4]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    write_pose_to_json(out_pkl, output_dir, thresh=0)

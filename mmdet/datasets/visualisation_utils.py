import warnings
import math

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import copy

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

from mmdet.datasets.kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, euler_angles_to_rotation_matrix, quaternion_to_euler_angle, draw_line, draw_points


def nms_with_IOU(bboxes_with_IOU, thresh=0.55):
    x1 = bboxes_with_IOU[:, 0]
    y1 = bboxes_with_IOU[:, 1]
    x2 = bboxes_with_IOU[:, 2]
    y2 = bboxes_with_IOU[:, 3]
    IOU_scores = bboxes_with_IOU[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = IOU_scores.argsort()[::-1]  ## indices stored

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_with_IOU_and_vote(bboxes_with_IOU, thresh=0.55, vote=0):
    x1 = bboxes_with_IOU[:, 0]
    y1 = bboxes_with_IOU[:, 1]
    x2 = bboxes_with_IOU[:, 2]
    y2 = bboxes_with_IOU[:, 3]
    # score = bboxes_with_IOU[:, 4]
    IOU_scores = bboxes_with_IOU[:, 5]
    model_type = bboxes_with_IOU[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = IOU_scores.argsort()[::-1]  ## indices stored

    keep = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        if vote > 0:
            remove_inds = np.where(ovr > thresh)[0]
            vote_index = np.append(model_type[order[remove_inds + 1]], model_type[i])
            if len(set(vote_index)) >= vote:
                keep.append(i)
        else:
            keep.append(i)

        order = order[inds + 1]

    return keep


def nms_with_IOU_and_vote_return_index(bboxes_with_IOU, thresh=0.55, vote=0):
    x1 = bboxes_with_IOU[:, 0]
    y1 = bboxes_with_IOU[:, 1]
    x2 = bboxes_with_IOU[:, 2]
    y2 = bboxes_with_IOU[:, 3]
    # score = bboxes_with_IOU[:, 4]
    IOU_scores = bboxes_with_IOU[:, 5]
    model_type = bboxes_with_IOU[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = IOU_scores.argsort()[::-1]  ## indices stored

    keep = {}
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        remove_inds = np.where(ovr > thresh)[0]

        if vote > 0:
            vote_index = np.append(model_type[order[remove_inds + 1]], model_type[i])
            if len(set(vote_index)) < vote:
                order = order[inds + 1]
                continue

        keep[i] = np.append(order[remove_inds + 1], i)
        order = order[inds + 1]
    return keep


def get_xy_from_z(boxes, t):
    boxes_copy = boxes.copy()
    x, y, z = t
    cx, cy = 1686.2379, 1354.9849
    fx, fy = 2304.5479, 2305.8757
    crop_top = 1480
    boxes_copy[1::2] += crop_top
    center = np.array([np.mean(boxes_copy[:-1][0::2]), np.mean(boxes_copy[1::2])])
    X = (center[0] - cx) * z / fx
    Y = (center[1] - cy) * z / fy

    # print('x,X,y,Y',x,X,y,Y)
    return np.array([X, Y, z])


def get_xy_from_z_mutually(boxes, t):
    boxes_copy = boxes.copy()
    x, y, z = t
    cx, cy = 1686.2379, 1354.9849
    fx, fy = 2304.5479, 2305.8757
    crop_top = 1480
    boxes_copy[1::2] += crop_top
    center = np.array([np.mean(boxes_copy[:-1][0::2]), np.mean(boxes_copy[1::2])])

    X = (center[0] - cx) * z / fx
    Y = (center[1] - cy) * z / fy
    Z1 = x * fx / (center[0] - cx)
    Z2 = y * fy / (center[1] - cy)
    # print('x,X,y,Y',x,X,y,Y)
    T = [np.array([X, Y, z]), np.array([x, y, Z1]), np.array([x, y, Z2])]
    return T


def restore_x_y_from_z(bboxes, trans_pred_world):
    # bboxes_refined = bboxes.copy()
    # print('bboxes',bboxes.shape)
    trans_pred_world_refinder = trans_pred_world.copy()
    for i in range(trans_pred_world.shape[0]):
        box = bboxes[i]
        t = trans_pred_world[i]
        X, Y = get_xy_from_z(box, t)
        trans_pred_world_refinder[i][0] = X
        trans_pred_world_refinder[i][1] = Y
    return trans_pred_world_refinder


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img_original = imread(img)
    img = img_original[1480:, :, :]

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    im_combime = img_original.copy()
    im_combime[1480:, :, :] = img
    if out_file is not None:
        imwrite(im_combime, out_file)


# TODO: merge this method with the one in BaseDetector
def show_result_kaggle_pku(img,
                           result,
                           class_names,
                           score_thr=0.3,
                           wait_time=0,
                           transparency=0.3,
                           show=True,
                           out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))

    img_original = imread(img)
    img = img_original[1480:, :, :]
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * (1 - transparency) + color_mask * transparency
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    im_combime = img_original.copy()
    im_combime[1480:, :, :] = img
    imshow_det_bboxes(
        im_combime,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


def get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names,
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


def refine_yaw_and_roll(img_original, bboxes, segms, class_names, euler_angle, quaternion_pred, trans_pred_world,
                        car_model_dict,
                        camera_matrix,
                        score_thr=0.1,
                        roll_threshold=0.2,
                        # roll_threshold = 0.18,
                        yaw_threshold=(0, 0.3)):
    ### we find that sometimes the predicted roll or yaw is out of normal range,so we confine it to normal range.
    ## roll mainly locates from -0.1 to 0.1 we confine the value out of absolute value of 0.2
    pi = math.pi
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
                yaw = 0.15  ## waited to be determined

            if yaw > yaw_threshold[1]:
                # print('yaw change',yaw,yaw_threshold[1])
                yaw = 0.15  ## waited to be determined

            if np.abs(roll - candidate_roll) > roll_threshold:
                # print('roll',roll,candidate_roll)

                roll = candidate_roll

            quaternion_refined = euler_angles_to_quaternions(np.array([yaw, pitch, roll]))
            quaternion_semisphere_refined = quaternion_upper_hemispher(quaternion_refined)
            quaternion_pred_refined[bbox_idx] = np.array(quaternion_semisphere_refined)
            flag = True

    return quaternion_pred_refined, flag


def restore_x_y_from_z_withIOU(img_original, bboxes, segms, class_names, euler_angle, trans_pred_world,
                               car_model_dict,
                               camera_matrix,
                               score_thr=0.1,
                               # refined_threshold1=5,
                               refined_threshold1=10,
                               refined_threshold2=28,
                               IOU_threshold=0.3):
    img = img_original[1480:, :, :].copy()

    trans_pred_world_refined = trans_pred_world.copy()
    for bbox_idx in range(len(bboxes)):
        if bboxes[bbox_idx, -1] <= score_thr:  ## we only restore case when score > score_thr(0.1)
            continue

        bbox = bboxes[bbox_idx]
        ## below is the predicted mask
        mask_all_pred = np.zeros(img.shape[:-1])  ## this is the background mask
        mask_all_mesh = np.zeros(img.shape[:-1])
        mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
        mask_all_pred += mask_pred
        mask_all_pred_area = np.sum(mask_all_pred == 1)

        t = trans_pred_world[bbox_idx]
        t_refined = get_xy_from_z(bbox, t)

        score_iou_mask_before, score_iou_before = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names,
                                                                mask_all_pred, mask_all_mesh, mask_all_pred_area,
                                                                euler_angle, t)
        score_iou_mask_after, score_iou_after = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names,
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


def restore_x_y_from_z_withIOU_mutual(img_original, bboxes, segms, class_names, euler_angle, trans_pred_world,
                                      car_model_dict,
                                      camera_matrix,
                                      score_thr=0.1, ):
    img = img_original[1480:, :, :].copy()
    trans_pred_world_refined = trans_pred_world.copy()
    for bbox_idx in range(len(bboxes)):
        if bboxes[bbox_idx, -1] <= score_thr:  ## we only restore case when score > score_thr(0.1)
            continue

        bbox = bboxes[bbox_idx]

        ## below is the predicted mask
        mask_all_pred = np.zeros(img.shape[:-1])  ## this is the background mask
        mask_all_mesh = np.zeros(img.shape[:-1])
        mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
        mask_all_pred += mask_pred
        mask_all_pred_area = np.sum(mask_all_pred == 1)

        t = trans_pred_world[bbox_idx]
        # t_refined = get_xy_from_z(bbox,t)
        T_refined = get_xy_from_z_mutually(bbox, t)

        _, score_iou_before = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names, mask_all_pred,
                                            mask_all_mesh, mask_all_pred_area, euler_angle, t)
        _, score_iou_after_1 = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names, mask_all_pred,
                                             mask_all_mesh, mask_all_pred_area, euler_angle, T_refined[0])
        _, score_iou_after_2 = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names, mask_all_pred,
                                             mask_all_mesh, mask_all_pred_area, euler_angle, T_refined[1])
        _, score_iou_after_3 = get_iou_score(bbox_idx, car_model_dict, camera_matrix, class_names, mask_all_pred,
                                             mask_all_mesh, mask_all_pred_area, euler_angle, T_refined[2])

        ## we find the highest score_iou_after
        score_concat = np.array([score_iou_after_1, score_iou_after_2, score_iou_after_3])
        idx_iou = np.argmax(score_concat)
        # print('idx_iou',idx_iou)

        score_tmp = copy.copy(score_iou_before)
        ## we only restore when iou increase 
        if score_concat[idx_iou] > score_iou_before + 0.05:
            # print('before after',score_iou_before,score_concat[idx_iou])
            trans_pred_world_refined[bbox_idx] = T_refined[idx_iou]
            score_tmp = score_concat[idx_iou]

    return trans_pred_world_refined


def get_IOU(img_original, bboxes, segms, six_dof, car_id2name,
            car_model_dict,
            unique_car_mode,
            camera_matrix):
    img = img_original[1480:, :, :].copy()
    bboxes_with_IOU = np.zeros((bboxes.shape[0], bboxes.shape[1] + 1)).astype(
        bboxes.dtype)  ## we add IOU score for each line

    quaternion_pred = six_dof['quaternion_pred']
    euler_angles = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
    car_cls_score_pred = six_dof['car_cls_score_pred']
    trans_pred_world = six_dof['trans_pred_world']
    car_labels = np.argmax(car_cls_score_pred, axis=1)
    kaggle_car_labels = [unique_car_mode[x] for x in car_labels]
    car_names = np.array([car_id2name[x].name for x in kaggle_car_labels])
    for bbox_idx in range(len(bboxes)):
        box = bboxes[bbox_idx]
        t = trans_pred_world[bbox_idx]
        ## below is the predicted mask
        mask_all_pred = np.zeros(img.shape[:-1])  ## this is the background mask
        mask_all_mesh = np.zeros(img.shape[:-1])
        mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
        mask_all_pred += mask_pred

        vertices = np.array(car_model_dict[car_names[bbox_idx]]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(car_model_dict[car_names[bbox_idx]]['faces']) - 1

        ea = euler_angles[bbox_idx]
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

        for tri in triangles:
            coord = np.array([img_cor_points[tri[0]][:2], img_cor_points[tri[1]][:2], img_cor_points[tri[2]][:2]],
                             dtype=np.int32)
            coord[:, 1] -= 1480
            cv2.drawContours(mask_all_mesh, np.int32([coord]), 0, 1, -1)

        intersection_area = np.sum(mask_all_pred * mask_all_mesh)
        union_area = np.sum(np.logical_or(mask_all_pred, mask_all_mesh))
        iou_score = intersection_area / union_area
        bboxes_with_IOU[bbox_idx] = np.append(box, iou_score)
    return bboxes_with_IOU


def draw_box_mesh_kaggle_pku(img_original, bboxes, segms, class_names,
                             car_model_dict,
                             camera_matrix,
                             trans_pred_world,
                             euler_angle,
                             score_thr=0.8,
                             thickness=1,
                             transparency=0.5,
                             font_scale=0.8,
                             ):
    img = img_original[1480:, :, :].copy()  ## crop half

    iou_flag = False
    trans_pred_world_raw = trans_pred_world.copy()
    if score_thr > 0:
        inds = bboxes[:, -1] > score_thr
        bboxes = bboxes[inds, :]
        segms = np.array(segms)[inds]
        trans_pred_world = trans_pred_world[inds, :]
        euler_angle = euler_angle[inds, :]
        class_names = class_names[inds]

    for bbox_idx in range(len(bboxes)):
        color_ndarray = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        color = tuple([int(i) for i in color_ndarray[0]])
        bbox = bboxes[bbox_idx]

        ## below is the predicted mask
        mask_all_pred = np.zeros(img.shape[:-1])  ## this is the background mask
        mask_all_mesh = np.zeros(img.shape[:-1])
        mask_pred = maskUtils.decode(segms[bbox_idx]).astype(np.bool)
        mask_all_pred += mask_pred
        mask_all_pred_area = np.sum(mask_all_pred == 1)
        # img[mask_pred] = img[mask_pred] * (1-transparency) + color_ndarray * transparency

        label_text = class_names[bbox_idx]
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        t = trans_pred_world[bbox_idx]

        ## time to draw mesh
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

        for tri in triangles:
            coord = np.array([img_cor_points[tri[0]][:2], img_cor_points[tri[1]][:2], img_cor_points[tri[2]][:2]],
                             dtype=np.int32)
            coord[:, 1] -= 1480
            cv2.polylines(img, np.int32([coord]), 1, color, thickness=1)
            cv2.drawContours(mask_all_mesh, np.int32([coord]), 0, 1, -1)
            # cv2.drawContours(img,np.int32([coord]),0,color,-1)

        intersection_area = np.sum(mask_all_pred * mask_all_mesh)
        union_area = np.sum(np.logical_or(mask_all_pred, mask_all_mesh))
        iou_mask_score = round(intersection_area / mask_all_pred_area, 3)
        iou_score = round(intersection_area / union_area, 3)
        label_text_t = ''
        cls_score = bboxes[bbox_idx][-1]

        if iou_score < 0.5:
            print('iou_score', iou_score, cls_score)

            iou_flag = True
        # for i in ea:
        #     i = round(i,4)
        #     label_text_t += str(i)
        #     label_text_t += ' '
        #
        # for i in t:
        #     i = round(i,4)
        #     label_text_t += str(i)
        #     label_text_t += ' '
        # label_text_t += str(iou_mask_score) + ' ' + str(iou_score) + ' ' + str(cls_score)
        label_text_t += str(iou_score) + ' ' + str(cls_score)
        cv2.rectangle(img, left_top, right_bottom, color, thickness=thickness)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text_t, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_ITALIC, font_scale, color)
    im_combime = img_original.copy()
    im_combime[1480:, :, :] = img
    return im_combime, iou_flag


def draw_result_kaggle_pku(img_original, bboxes, segms, car_names,
                           car_model_dict,
                           camera_matrix,
                           trans_pred_world,
                           euler_angle,
                           score_thr=0.1,
                           transparency=0.5,
                           ):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    img = img_original[1480:, :, :]
    color_lists = []
    # draw segmentation masks
    if segms is not None:
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_lists.append(color_mask)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * (1 - transparency) + color_mask * transparency
    # draw bounding boxes
    im_combime = img_original.copy()
    im_combime[1480:, :, :] = img
    im_combime = imdraw_det_bboxes(
        im_combime,
        bboxes,
        car_names,
        car_model_dict,
        camera_matrix,
        trans_pred_world,
        euler_angle,
        color_lists,
        score_thr=score_thr)

    return im_combime


def imdraw_det_bboxes(img,
                      bboxes,
                      class_names,
                      car_model_dict,
                      camera_matrix,
                      trans_pred_world,
                      euler_angle,
                      color_lists,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img_original = img.copy()
    img = img_original[1480:, :, :]

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        trans_pred_world = trans_pred_world[inds, :]
        euler_angle = euler_angle[inds, :]
    assert len(bboxes) == len(trans_pred_world) == len(euler_angle)
    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox_idx in range(len(bboxes)):
        bbox = bboxes[bbox_idx]
        label_text = class_names[bbox_idx]
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        # now we draw mesh
        vertices = np.array(car_model_dict[class_names[bbox_idx]]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(car_model_dict[class_names[bbox_idx]]['faces']) - 1

        t = trans_pred_world[bbox_idx]
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

        color_mesh = np.int32(color_lists[bbox_idx][0])
        color_tuple = tuple([int(x) for x in color_mesh])
        for t in triangles:
            coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                             dtype=np.int32)
            # This will draw the mask for segmenation
            # cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
            coord[:, 1] -= 1480
            cv2.polylines(img, np.int32([coord]), 1, color=color_tuple)

    im_combime = img_original.copy()
    im_combime[1480:, :, :] = img
    return im_combime


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))


def visual_PnP(img, PnP_pred, camera_matrix, vertices, triangles):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """

    for pcar_idx in range(len(PnP_pred)):
        # now we draw mesh
        pcar = PnP_pred[pcar_idx]
        t = pcar['x'], pcar['y'], pcar['z']
        yaw, pitch, roll = pcar['yaw'], pcar['pitch'], pcar['roll']
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

        color_mesh = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

        color_tuple = tuple([int(x) for x in color_mesh[0]])
        for t in triangles:
            coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                             dtype=np.int32)
            cv2.polylines(img, np.int32([coord]), 1, color=color_tuple)

    return img

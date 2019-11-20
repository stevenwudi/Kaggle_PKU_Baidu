import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import numpy as np

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

from mmdet.datasets.kaggle_pku_utils import euler_to_Rot, euler_angles_to_quaternions, \
    quaternion_upper_hemispher, euler_angles_to_rotation_matrix, quaternion_to_euler_angle, draw_line, draw_points


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
            img[mask] = img[mask] * (1-transparency) + color_mask * transparency
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


def draw_result_kaggle_pku(img_original, bboxes, segms, car_names,
                           car_model_dict,
                           camera_matrix,
                           trans_pred_world,
                           euler_angle,
                           score_thr=0.1,
                           transparency=0.3,
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
            img[mask] = img[mask] * (1-transparency) + color_mask * transparency
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
            coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]], dtype=np.int32)
            # This will draw the mask for segmenation
            #cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
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

"""
    Brief: Utility functions of apolloscape tool kit
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from math import sin, cos
import os
from pycocotools import mask as maskUtils


def mesh_point_to_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax


def euler_angles_to_quaternions(angle):
    """
    Convert euler angels to quaternions representation.
    该公式适用的yaw, pitch, roll与label里的定义不一样，需要做相应的变换 yaw, pitch, roll => pitch, yaw, roll

    Input:
        angle: n x 3 matrix, each row is [yaw, pitch, roll]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]

    # yaw, pitch, roll => pitch, yaw, roll
    pitch, yaw, roll = angle[:, 0], angle[:, 1], angle[:, 2]

    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    if in_dim == 1:
        return q[0]
    return q


def euler_angles_to_quaternions_apollo(angle):
    """Convert euler angels to quaternions representation.
    Input:
        angle: n x 3 matrix, each row is [roll, pitch, yaw]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = angle[:, 0], angle[:, 1], angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp
    if in_dim == 1:
        return q[0]
    return q


def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 1 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                print(q)
                q[3] = 1

    return q


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


def quaternion_to_euler_angle_apollo(q):
    """Convert quaternion to euler angel.
    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [roll, pitch, yaw]
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

    return roll, pitch, yaw


def intrinsic_vec_to_mat(intrinsic, shape=None):
    """Convert a 4 dim intrinsic vector to a 3x3 intrinsic
       matrix
    """
    if shape is None:
        shape = [1, 1]

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = intrinsic[0] * shape[1]
    K[1, 1] = intrinsic[1] * shape[0]
    K[0, 2] = intrinsic[2] * shape[1]
    K[1, 2] = intrinsic[3] * shape[0]
    K[2, 2] = 1.0

    return K


def round_prop_to(num, base=4.):
    """round a number to integer while being propotion to
       a given base number
    """
    return np.ceil(num / base) * base


def euler_to_Rot_YPR(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(

                      roll), cos(roll), 0],
                  [0, 0, 1]])

    return Y, P, R


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


def euler_to_Rot_apollo(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])

    return np.dot(np.dot(R, Y), P)


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


def rotation_matrix_to_euler_angles(R, check=True):
    """Convert rotation matrix to euler angles
    Input:
        R: 3 x 3 rotation matrix
        check: whether Checeuler_angles_to_quaternionsk if a matrix is a valid
            rotation matrix.
    Output:
        euler angle [x/roll, y/pitch, z/yaw]
    """

    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        # return n < 3 *(1e-6)
        # Di Wu relax the condition for TLESS dataset
        return n < 1e-5

    if check:
        assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rot2eul(R, euler_original, thresh=1e-5, debug=False):
    """
    https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
    How to handle case for cos\theta == 0:
    https://www.gregslabaugh.net/publications/euler.pdf
    :param R:
    :return:
    """
    if R[2, 0]-1 > thresh or np.abs(R[2, 0]-1) < thresh:
        beta = -np.pi/2
        alpha = np.arctan2(-R[0, 1], -R[0, 2]) - euler_original[2]
        gamma = euler_original[2]
        if debug:
            print(beta)
    elif R[2, 0] - (-1) < thresh or np.abs(R[2, 0]+1) < thresh:
        beta = np.pi/2
        alpha = np.arctan2(R[0, 1], R[0, 2]) + euler_original[2]
        gamma = euler_original[2]
        if debug:
            print(beta)
    else:
        beta = -np.arcsin(R[2, 0])
        alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
        gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return -np.array((beta, alpha, gamma))


def convert_pose_mat_to_6dof(pose_file_in, pose_file_out):
    """Convert a pose file with 4x4 pose mat to 6 dof [xyz, rot]
    representation.
    Input:
        pose_file_in: a pose file with each line a 4x4 pose mat
        pose_file_out: output file save the converted results
    """

    poses = [line for line in open(pose_file_in)]
    output_motion = np.zeros((len(poses), 6))
    f = open(pose_file_out, 'w')
    for i, line in enumerate(poses):
        nums = line.split(' ')
        mat = [np.float32(num.strip()) for num in nums[:-1]]
        image_name = nums[-1].strip()
        mat = np.array(mat).reshape((4, 4))

        xyz = mat[:3, 3]
        rpy = rotation_matrix_to_euler_angles(mat[:3, :3])
        output_motion = np.hstack((xyz, rpy)).flatten()
        out_str = '%s %s\n' % (image_name, np.array2string(output_motion,
                                                           separator=',',
                                                           formatter={'float_kind': lambda x: "%.7f" % x})[1:-1])
        f.write(out_str)
    f.close()

    return output_motion


def trans_vec_to_mat(rot, trans, dim=4):
    """ project vetices based on extrinsic parameters
    """
    mat = euler_angles_to_rotation_matrix(rot)
    mat = np.hstack([mat, trans.reshape((3, 1))])
    if dim == 4:
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat


def project(pose, scale, vertices):
    """ transform the vertices of a 3D car model based on labelled pose
    Input:
        pose: 0-3 rotation, 4-6 translation
        scale: the scale at each axis of the car
        vertices: the vertices position
    """

    if np.ndim(pose) == 1:
        mat = trans_vec_to_mat(pose[:3], pose[3:])
    elif np.ndim(pose) == 2:
        mat = pose

    vertices = vertices * scale
    p_num = vertices.shape[0]

    points = vertices.copy()
    points = np.hstack([points, np.ones((p_num, 1))])
    points = np.matmul(points, mat.transpose())

    return points[:, :3]


def plot_images(images,
                layout=[2, 2],
                fig_size=10,
                save_fig=False,
                fig_name=None):
    """Plot a dictionary of images:
    Input:
        images: dictionary {'image', image}
        layout: the subplot layout of output
        fig_size: size of figure
        save_fig: bool, whether save the plot images
        fig_name: if save_fig, then provide a name to save
    """

    plt.figure(figsize=(10, 5))
    pylab.rcParams['figure.figsize'] = fig_size, fig_size / 2
    Keys = images.keys()
    for iimg, name in enumerate(Keys):
        assert len(images[name].shape) >= 2

    for iimg, name in enumerate(Keys):
        s = plt.subplot(layout[0], layout[1], iimg + 1)
        plt.imshow(images[name])

        s.set_xticklabels([])
        s.set_yticklabels([])
        s.set_title(name)
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')

    plt.tight_layout()
    if save_fig:
        pylab.savefig(fig_name)
    else:
        plt.show()


def extract_intrinsic(dataset):
    intrinsic_mat = dataset.Car3D.get_intrinsic_mat()
    fx = intrinsic_mat[0][0]
    fy = intrinsic_mat[1][1]
    cx = intrinsic_mat[0][2]
    cy = intrinsic_mat[1][2]
    return fx, fy, cx, cy


def im_car_trans_geometric(dataset, boxes, euler_angle, car_cls, im_scale=1.0):
    ###
    fx, fy, cx, cy = extract_intrinsic(dataset)

    car_cls_max = np.argmax(car_cls, axis=1)
    car_names = [dataset.Car3D.car_id2name[x].name for x in dataset.Car3D.unique_car_models[car_cls_max]]

    if im_scale != 1:
        raise Exception("not implemented, check it")
    boxes = boxes / im_scale

    car_trans_pred = []
    for car_idx in range(boxes.shape[0]):
        box = boxes[car_idx]
        xc = ((box[0] + box[2]) / 2 - cx) / fx
        yc = ((box[1] + box[3]) / 2 - cy) / fy
        ymax = (box[3] - cy) / fy

        # project 3D points to 2d image plane
        euler_angle_i = euler_angle[car_idx]
        rmat = euler_angles_to_rotation_matrix(euler_angle_i)

        car = dataset.Car3D.car_models[car_names[car_idx]]
        x_y_z_R = np.matmul(rmat, np.transpose(np.float32(car['vertices'])))
        Rymax = x_y_z_R[1, :].max()
        Rxc = x_y_z_R[0, :].mean()
        Ryc = x_y_z_R[1, :].mean()
        Rzc = x_y_z_R[2, :].mean()
        zc = (Ryc - Rymax) / (yc - ymax)

        xt = zc * xc - Rxc
        yt = zc * yc - Ryc
        zt = zc - Rzc
        pred_pose = np.array([xt, yt, zt])
        car_trans_pred.append(pred_pose)

    return np.array(car_trans_pred)


def im_car_trans_geometric_ssd6d(dataset, boxes, euler_angle, car_cls, im_scale=1.0):
    ###
    fx, fy, cx, cy = extract_intrinsic(dataset)

    car_cls_max = np.argmax(car_cls, axis=1)
    car_names = [dataset.Car3D.car_id2name[x].name for x in dataset.Car3D.unique_car_models[car_cls_max]]

    if im_scale != 1:
        raise Exception("not implemented, check it")
    boxes = boxes / im_scale

    car_trans_pred = []
    # canonical centroid zr = 10.0
    zr = 10.0
    trans_vect = np.zeros((3, 1))
    trans_vect[2] = zr
    for car_idx in range(boxes.shape[0]):
        box = boxes[car_idx]

        # lr denotes diagonal length of the precomputed bounding box and ls denotes the diagonal length
        # of the predicted bounding box on the image plane
        ls = np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2)
        # project 3D points to 2d image plane
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        euler_angle_i = euler_angle[car_idx]
        rmat = euler_angles_to_rotation_matrix(euler_angle_i)
        car = dataset.Car3D.car_models[car_names[car_idx]]
        x_y_z_R = np.matmul(rmat, np.transpose(np.float32(car['vertices'])))
        x_y_z_R_T = x_y_z_R + trans_vect
        x_y_z_R_T_hat = x_y_z_R_T / x_y_z_R_T[2, :]

        u = fx * x_y_z_R_T_hat[0, :] + cx
        v = fy * x_y_z_R_T_hat[1, :] + cy
        lr = np.sqrt((u.max() - u.min()) ** 2 + (v.max() - v.min()) ** 2)

        zs = lr * zr / ls

        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2
        xc_syn = (u.max() + u.min()) / 2
        yc_syn = (v.max() + v.min()) / 2

        xt = zs * (xc - xc_syn) / fx
        yt = zs * (yc - yc_syn) / fy

        pred_pose = np.array([xt, yt, zs])
        car_trans_pred.append(pred_pose)

    return np.array(car_trans_pred)


def draw_line(image, points):
    color = (0, 0, 255)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 8)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 8)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 8)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 8)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    #         if p_x > image.shape[1] or p_y > image.shape[0]:
    #             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image


def filter_igore_masked_images(
        img_name,
        mask_list,
        img_prefix,
        iou_threshold=0.5):
    """
    We filter out the ignore mask according to IoU
    :param mask_list:
    :param img_prefix:
    :return:
    """
    # a hard coded path for extractin ignore test mask region
    if 'valid' in img_prefix:
        mask_dir = img_prefix.replace('validation_images', 'train_masks')
    else:
        mask_dir = img_prefix.replace('test_images', 'test_masks')

    mask_file = os.path.join(mask_dir, img_name + '.jpg')
    if os.path.isfile(mask_file):
        mask_im = cv2.imread(mask_file)
        mask_im = np.mean(mask_im, axis=2)
        mask_im[mask_im > 0] = 1
    else:
        # there is no ignore mask
        return [True] * len(mask_list)

    idx_keep_mask = [False] * len(mask_list)
    for i, mask_car_rle in enumerate(mask_list):
        mask_car = maskUtils.decode(mask_car_rle)
        im_combime = np.zeros(mask_im.shape)
        im_combime[1480:, :] = mask_car

        # now we calculate the IoU:
        area_car = im_combime.sum()
        interception = im_combime * mask_im
        area_interception = interception.sum()
        iou_car = area_interception / area_car
        if iou_car < iou_threshold:
            idx_keep_mask[i] = True

    return idx_keep_mask


def filter_igore_masked_using_RT(
        img_name,
        six_dof,
        img_prefix,
        dataset,
        iou_threshold=0.9):
    """
    We filter out the ignore mask according to IoU
    :param mask_list:
    :param img_prefix:
    :return:
    """

    # a hard coded path for extractin ignore test mask region
    if 'valid' in img_prefix:
        mask_dir = img_prefix.replace('validation_images', 'train_masks')
    else:
        mask_dir = img_prefix.replace('test_images', 'test_masks')

    mask_file = os.path.join(mask_dir, img_name + '.jpg')
    if os.path.isfile(mask_file):
        mask_im = cv2.imread(mask_file)
        mask_im = np.mean(mask_im, axis=2)
        mask_im[mask_im > 0] = 1
    else:
        # there is no ignore mask
        return [True] * six_dof['quaternion_pred'].shape[0]

    idx_keep_mask = [False] * six_dof['quaternion_pred'].shape[0]

    # output is a tuple of three elements
    car_cls_score_pred = six_dof['car_cls_score_pred']
    quaternion_pred = six_dof['quaternion_pred']
    trans_pred_world = six_dof['trans_pred_world']
    euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
    car_labels = np.argmax(car_cls_score_pred, axis=1)
    kaggle_car_labels = [dataset.unique_car_mode[x] for x in car_labels]
    car_names = [dataset.car_id2name[x].name for x in kaggle_car_labels]

    for i in range(len(car_cls_score_pred)):

        # We start to render the mask according to R,T
        # now we draw mesh
        # car_id2name is from:
        # https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/car_models.py
        car_name = car_names[i]
        vertices = np.array(dataset.car_model_dict[car_name]['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(dataset.car_model_dict[car_name]['faces']) - 1

        # project 3D points to 2d image plane
        yaw, pitch, roll = euler_angle[i]
        # I think the pitch and yaw should be exchanged
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = np.array(trans_pred_world[i])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(dataset.camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        # project 3D points to 2d image plane
        mask_seg = np.zeros(dataset.image_shape, dtype=np.uint8)
        for t in triangles:
            coord = np.array([img_cor_points[t[0]][:2], img_cor_points[t[1]][:2], img_cor_points[t[2]][:2]],
                             dtype=np.int32)
            # This will draw the mask for segmenation
            cv2.drawContours(mask_seg, np.int32([coord]), 0, (255, 255, 255), -1)
            # cv2.polylines(mask_seg_mesh, np.int32([coord]), 1, (0, 255, 0))

        # now we calculate the IoU:
        area_car = mask_seg.sum()
        interception = mask_seg * mask_im
        area_interception = interception.sum()
        iou_car = area_interception / area_car
        if iou_car < iou_threshold:
            idx_keep_mask[i] = True
        # else:
        #     # iou car, we save it
        #     img_output_dir = '/data/Kaggle/wudi_data/work_dirs/filter_image_mask_demo'
        #     im_name = os.path.join(img_output_dir, img_name + '_%d.jpg'%i)
        #     im_combined = mask_im*0.5 + mask_seg/255*0.5
        #     imwrite(im_combined*255, im_name)

    return idx_keep_mask


def coords2str(coords):
    s = []
    for c in coords:
        for l in c:
            s.append('%.5f' % l)
    return ' '.join(s)


def filter_output(output_idx, outputs, conf_thresh, img_prefix, dataset):
    output = outputs[output_idx]
    file_name = os.path.basename(output[2]["file_name"])
    ImageId = ".".join(file_name.split(".")[:-1])
    CAR_IDX = 2  # this is the coco car class

    # Wudi change the conf to car prediction
    if len(output[0][CAR_IDX]):
        conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox
        idx_conf = conf > conf_thresh

        # this filtering step will takes 2 second per iterations
        # idx_keep_mask = filter_igore_masked_images(ImageId[idx_img], output[1][CAR_IDX], img_prefix)
        idx_keep_mask = filter_igore_masked_using_RT(ImageId, output[2], img_prefix, dataset)
        # the final id should require both
        idx = idx_conf * idx_keep_mask
        if 'euler_angle' in output[2].keys():
            euler_angle = output[2]['euler_angle']
        else:
            euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])
        # This is a new modification because in CYH's new json file;
        translation = output[2]['trans_pred_world']
        coords = np.hstack((euler_angle[idx], translation[idx], conf[idx, None]))
        coords_str = coords2str(coords)
    else:
        coords_str = ""

    return coords_str, ImageId


def non_max_suppression_fast(boxes, overlapThresh):
    """
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes:
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    # initialize the list of picked indexes
    pick = []

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


if __name__ == '__main__':
    pass

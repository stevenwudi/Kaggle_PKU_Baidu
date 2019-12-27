"""
Finding camera parameters R, T
"""
import mmcv
import os
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imsave
import tqdm
import pycocotools.mask as maskUtils

import neural_renderer as nr

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle, euler_to_Rot, rotation_matrix_to_euler_angles

from mmdet.datasets.car_models import car_id2name


class Model(nn.Module):
    def __init__(self,
                 vertices,
                 faces,
                 Rotation_Matrix,
                 T,
                 eular_angle,
                 mask_full_size,
                 camera_matrix,
                 image_size,
                 iou_threshold=0.9,
                 fix_rot=False):
        super(Model, self).__init__()

        vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        self.translation_original = T
        self.euler_original = eular_angle
        # create textures
        texture_size = 1
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # we set the loss threshold to stop perturbation
        self.mask_full_size = mask_full_size
        self.mask_sum = self.mask_full_size.sum()
        self.loss_thresh = iou_threshold

        image_ref = torch.from_numpy(mask_full_size.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.register_buffer('K', torch.from_numpy(camera_matrix))
        # setup renderer
        if fix_rot:
            R = torch.from_numpy(np.array(Rotation_Matrix[None, :], dtype=np.float32)).cuda()
            self.register_buffer('R', R)

            renderer = nr.Renderer(image_size=image_size,
                                   orig_size=image_size,
                                   camera_mode='projection',
                                   K=camera_matrix[None, :, :],
                                   R=R)

        if not fix_rot:
            renderer = nr.Renderer(image_size=image_size,
                                   orig_size=image_size,
                                   camera_mode='projection',
                                   K=camera_matrix[None, :, :])
            renderer.R = nn.Parameter(torch.from_numpy(np.array(Rotation_Matrix[None, :], dtype=np.float32)))
        renderer.t = nn.Parameter(torch.from_numpy(np.array(T[None, :], dtype=np.float32)))
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        interception = torch.sum(torch.abs(image * self.image_ref[None, :, :]))
        union = torch.sum(image) + torch.sum(self.image_ref) - interception
        loss = - interception / union
        # loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()

    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)


def get_updated_RT(vertices,
                   faces,
                   Rotation_Matrix,
                   T,
                   eular_angle,
                   mask_full_size,
                   camera_matrix,
                   image_size,
                   iou_threshold,
                   num_epochs=100,
                   draw_flag=False,
                   lr=0.05):
    model = Model(vertices,
                  faces,
                  Rotation_Matrix,
                  T,
                  eular_angle,
                  mask_full_size,
                  camera_matrix,
                  image_size=image_size,
                  iou_threshold=iou_threshold)
    if False:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loop = tqdm.tqdm(range(num_epochs))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if draw_flag:  # We don't save the images
            images = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
            image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
            imsave('/tmp/_tmp_%04d.png' % i, image)
        loop.set_description('Optimizing (loss %.4f)' % (loss.data))
        ### we print some updates
        if True:
            updated_translation = model.renderer.t.detach().cpu().numpy()[0]
            original_translation = model.translation_original

            updated_rot_matrix = model.renderer.R.detach().cpu().numpy()[0]
            updated_euler_angle = rotation_matrix_to_euler_angles(updated_rot_matrix, check=False)
            original_euler_angle = model.euler_original
            print('Origin translation - > updated tranlsation')
            print(original_translation)
            print(updated_translation)
            print('Origin eular angle - > updated eular angle')
            print(original_euler_angle)
            print(updated_euler_angle)
        if loss.item() < -model.loss_thresh:
            break

    updated_translation = model.renderer.t.detach().cpu().numpy()[0]
    updated_rot_matrix = model.renderer.R.detach().cpu().numpy()[0]

    ea = rotation_matrix_to_euler_angles(updated_rot_matrix, check=False)
    ea[0], ea[1], ea[2] = -ea[1], -ea[0], -ea[2]
    return updated_translation, ea


def finetune_RT(outputs,
                dataset,
                iou_threshold=0.9,
                num_epochs=50,
                draw_flag=False,
                # lr=0.05,
                lr=0.1,
                conf_thresh=0.1,
                tmp_save_dir='/data/Kaggle/wudi_data/tmp_output'):
    """

    :param outputs:
    :param dataset:
    :param difference_ratio:
    :return:
    """
    CAR_IDX = 2
    outputs_update = outputs.copy()
    for img_idx in range(len(outputs)):
        output = outputs[img_idx]
        bboxes, segms, six_dof = output[0], output[1], output[2]
        car_cls_score_pred = six_dof['car_cls_score_pred']
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world']
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        kaggle_car_labels = [dataset.unique_car_mode[x] for x in car_labels]
        car_names = [car_id2name[x].name for x in kaggle_car_labels]
        euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])

        conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox
        idx_conf = conf > conf_thresh

        # The following code is highly independent and parrallisable,
        # CYH help to parralise the following code

        for car_idx in range(len(quaternion_pred)):
            if idx_conf[car_idx]:  # we only update conf threshold larger than 0.1
                # The the HTC predicted Mask which is served as the GT Mask
                segms_car = segms[CAR_IDX][car_idx]
                mask = maskUtils.decode(segms_car)
                mask_full_size = np.zeros((2710, 3384))
                mask_full_size[1480:, :] = mask
                # Get car mesh--> vertices and faces
                car_name = car_names[car_idx]
                vertices = np.array(dataset.car_model_dict[car_name]['vertices'])
                vertices[:, 1] = -vertices[:, 1]
                faces = np.array(dataset.car_model_dict[car_name]['faces']) - 1
                # Get prediction of Rotation Matrix and  Translation
                ea = euler_angle[car_idx]
                yaw, pitch, roll = ea[0], ea[1], ea[2]
                yaw, pitch, roll = -pitch, -yaw, -roll
                Rotation_Matrix = euler_to_Rot(yaw, pitch, roll).T
                T = trans_pred_world[car_idx]

                T_update, R_update = get_updated_RT(vertices,
                                                    faces,
                                                    Rotation_Matrix,
                                                    T,
                                                    [yaw, pitch, roll],
                                                    mask_full_size,
                                                    dataset.camera_matrix,
                                                    image_size=(3384, 2710),
                                                    iou_threshold=iou_threshold,
                                                    num_epochs=num_epochs,
                                                    draw_flag=draw_flag,
                                                    lr=lr)
                if 'eular_angle' in outputs_update[img_idx][2]:
                    outputs_update[img_idx][2]['eular_angle'].append(R_update)
                else:
                    outputs_update[img_idx][2]['eular_angle'] = []
                    outputs_update[img_idx][2]['eular_angle'].append(R_update)
                outputs_update[img_idx][2]['trans_pred_world'][car_idx] = T_update

        outputs_update[img_idx][0] = bboxes[idx_conf]
        outputs_update[img_idx][1] = segms[idx_conf]
        outputs_update[img_idx][2]['eular_angle'] = np.array(outputs_update[img_idx][2]['eular_angle'])

        # We will save a picke file here because every image takes time and it could break
        # one image will typically takes 5 (epoch/it) * 50 (epochs) * 20 (num_cars) = 5000 second
        tmp_save_dir = '/data/Kaggle/wudi_data/tmp_output/'

        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        output_name = tmp_save_dir + 'output_%04d.pkl' % img_idx
        mmcv.dump(outputs_update[img_idx], output_name)
    return outputs_update

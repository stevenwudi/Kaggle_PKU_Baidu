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
from scipy.spatial.transform import Rotation as R
import neural_renderer as nr

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle, euler_to_Rot, rotation_matrix_to_euler_angles

from mmdet.datasets.car_models import car_id2name
from mmdet.utils import RotationDistance, TranslationDistance
import imageio
import glob


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


class Model(nn.Module):
    def __init__(self,
                 vertices,
                 faces,
                 Rotation_Matrix,
                 T,
                 euler_angle,
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
        self.euler_original = euler_angle
        self.fix_rot = fix_rot
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
                   euler_angle,
                   mask_full_size,
                   camera_matrix,
                   image_size,
                   iou_threshold,
                   num_epochs=50,
                   draw_flag=False,
                   output_gif=None,
                   lr=0.05,
                   fix_rot=False):
    model = Model(vertices,
                  faces,
                  Rotation_Matrix,
                  T,
                  euler_angle,
                  mask_full_size,
                  camera_matrix,
                  image_size=image_size,
                  iou_threshold=iou_threshold,
                  fix_rot=fix_rot)
    if draw_flag:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loop = tqdm.tqdm(range(num_epochs))
    # for i in loop:
    for i in range(num_epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if draw_flag:  # We don't save the images
            images = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
            image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
            image[:, :, 1] += model.image_ref.detach().cpu().numpy()
            imsave('/tmp/_tmp_%04d.png' % i, image)
            ### we print some updates
            print('Optimizing (loss %.4f)' % (loss.data))
            updated_translation = model.renderer.t.detach().cpu().numpy()[0]
            original_translation = model.translation_original
            changed_dis = TranslationDistance(original_translation, updated_translation, abs_dist=True)
            print('Origin translation: %s - > updated tranlsation: %s. Changed distance: %.4f'
                  % (
                  np.array2string(np.array(original_translation)), np.array2string(updated_translation), changed_dis))
            if not fix_rot:
                updated_rot_matrix = model.renderer.R.detach().cpu().numpy()[0]
                updated_euler_angle = rotation_matrix_to_euler_angles(updated_rot_matrix, check=False)
                original_euler_angle = rotation_matrix_to_euler_angles(Rotation_Matrix, check=False)

                changed_rot = RotationDistance(original_euler_angle, updated_euler_angle)

                print('Origin eular angle: %s - > updated eular angle: %s. Changed distance: %4.f'
                      % (np.array2string(np.array(original_euler_angle)), np.array2string(updated_euler_angle),
                         changed_rot))

        if loss.item() < -model.loss_thresh:
            break

    updated_translation = model.renderer.t.detach().cpu().numpy()[0]
    updated_rot_matrix = model.renderer.R.detach().cpu().numpy()[0]
    if draw_flag:
        make_gif(output_gif)
    # ea = rotation_matrix_to_euler_angles(updated_rot_matrix, check=False)
    # ea[0], ea[1], ea[2] = -ea[1], -ea[0], -ea[2]
    return updated_translation, updated_rot_matrix


def finetune_RT(outputs,
                dataset,
                iou_threshold=0.8,
                num_epochs=50,
                draw_flag=True,
                lr=0.1,  # lr=0.05,
                conf_thresh=0.1,
                tmp_save_dir='/data/Kaggle/wudi_data/tmp_output/',
                fix_rot=False):
    """
    :param outputs:
    :param dataset:
    :param difference_ratio:
    :return:
    """
    CAR_IDX = 2
    outputs_update = outputs.copy()
    for img_idx in tqdm.tqdm(range(len(outputs))):
        output = outputs[img_idx]
        bboxes, segms, six_dof = output[0], output[1], output[2]
        car_cls_score_pred = six_dof['car_cls_score_pred']
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world']
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        kaggle_car_labels = [dataset.unique_car_mode[x] for x in car_labels]
        car_names = [car_id2name[x].name for x in kaggle_car_labels]
        euler_angles = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])

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
                ea = euler_angles[car_idx]
                yaw, pitch, roll = ea[0], ea[1], ea[2]
                yaw, pitch, roll = -pitch, -yaw, -roll
                Rotation_Matrix = euler_to_Rot(yaw, pitch, roll).T
                T = trans_pred_world[car_idx]

                if draw_flag:
                    output_gif = tmp_save_dir + '/' + output[2]['file_name'].split('/')[-1][:-4] + '_' + str(
                        car_idx) + '.gif'
                else:
                    output_gif = None
                T_update, updated_rot_matrix = get_updated_RT(vertices,
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
                                                              output_gif=output_gif,
                                                              lr=lr,
                                                              fix_rot=fix_rot)
                if fix_rot:
                    R_update = ea  # we don't change the euler angle here
                else:
                    R_update = rotation_matrix_to_euler_angles(updated_rot_matrix)
                if 'euler_angle' in outputs_update[img_idx][2]:
                    outputs_update[img_idx][2]['euler_angle'].append(R_update)
                else:
                    outputs_update[img_idx][2]['euler_angle'] = []
                    outputs_update[img_idx][2]['euler_angle'].append(R_update)
                outputs_update[img_idx][2]['trans_pred_world'][car_idx] = T_update

            # We still need  to update so the car_idx does not break
            if 'euler_angle' in outputs_update[img_idx][2]:
                outputs_update[img_idx][2]['euler_angle'].append(euler_angles[car_idx])
            else:
                outputs_update[img_idx][2]['euler_angle'] = []
                outputs_update[img_idx][2]['euler_angle'].append(euler_angles[car_idx])

        outputs_update[img_idx][2]['euler_angle'] = np.array(outputs_update[img_idx][2]['euler_angle'])

        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        output_name = tmp_save_dir + '/' + output[2]['file_name'].split('/')[-1][:-4] + '.pkl'
        mmcv.dump(outputs_update[img_idx], output_name)
    return outputs_update

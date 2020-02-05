"""
Finding camera parameters R, T
Image per batch --> this is not likely to work!!!
"""
import mmcv
from mmcv import imwrite, imread
from skimage import color
import shutil
import glob
import os
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imsave
import pycocotools.mask as maskUtils
import neural_renderer as nr

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle, euler_to_Rot, rot2eul

from mmdet.datasets.car_models import car_id2name
from mmdet.utils import RotationDistance, TranslationDistance
import imageio


class Model(nn.Module):
    def __init__(self,
                 vertices,
                 faces,
                 Rotation_Matrix,
                 T,
                 euler_angle,
                 mask_full_size,
                 masked_grayscale_img,
                 camera_matrix,
                 image_size,
                 loss_thresh=0.9,
                 fix_rot=False,
                 fix_trans=False,
                 fix_light_source=True,
                 light_intensity_directional=0.1,
                 light_intensity_ambient=0.1,
                 light_direction=[1, -2, 1]
                 ):
        super(Model, self).__init__()

        vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.translation_original = T
        self.euler_original = euler_angle
        self.fix_rot = fix_rot
        self.fix_trans = fix_trans
        self.fix_light_source = fix_light_source
        # create textures
        texture_size = 1
        textures = torch.ones(T.shape[0], self.faces.shape[1], texture_size, texture_size, texture_size, 3,
                              dtype=torch.float32)
        self.register_buffer('textures', textures)

        # we set the loss threshold to stop perturbation
        self.mask_full_size = mask_full_size
        self.mask_sum = self.mask_full_size.sum()
        self.loss_thresh = -loss_thresh

        image_ref = torch.from_numpy(mask_full_size.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        masked_grayscale_img = torch.from_numpy(masked_grayscale_img.astype(np.float32))
        self.register_buffer('masked_grayscale_img', masked_grayscale_img)

        # camera parameters
        self.register_buffer('K', torch.from_numpy(camera_matrix))

        # initialise the renderer
        renderer = nr.Renderer(image_size=image_size,
                               orig_size=image_size,
                               camera_mode='projection',
                               K=camera_matrix[None, :, :])

        if fix_rot:
            R = torch.from_numpy(np.array(Rotation_Matrix, dtype=np.float32)).cuda()
            self.register_buffer('R', R)
            renderer.R = R
        else:
            renderer.R = nn.Parameter(torch.from_numpy(np.array(Rotation_Matrix, dtype=np.float32)))

        if fix_trans:
            t = torch.from_numpy(np.array(T, dtype=np.float32)).cuda()
            self.register_buffer('t', t)
            renderer.t = t
        else:
            renderer.t = nn.Parameter(torch.from_numpy(np.array(T, dtype=np.float32)))

        if fix_light_source:
            renderer.light_intensity_directional = torch.from_numpy(
                np.array(light_intensity_directional, dtype=np.float32)).cuda()
            renderer.light_intensity_ambient = torch.from_numpy(
                np.array(light_intensity_ambient, dtype=np.float32)).cuda()
            renderer.light_direction = torch.from_numpy(np.array(light_direction, dtype=np.float32)).cuda()
        else:
            renderer.light_intensity_directional = nn.Parameter(
                torch.from_numpy(np.array(light_intensity_directional, dtype=np.float32)))
            renderer.light_intensity_ambient = nn.Parameter(
                torch.from_numpy(np.array(light_intensity_ambient, dtype=np.float32)))
            renderer.light_direction = nn.Parameter(torch.from_numpy(np.array(light_direction, dtype=np.float32)))
        self.renderer = renderer

    def forward(self):
        image_rgb = self.renderer(self.vertices, self.faces, self.textures, mode='rgb')
        image_gray = image_rgb.sum(dim=0).sum(dim=0)
        image_gray = image_gray /image_gray.max()
        loss = torch.sum((image_gray - self.masked_grayscale_img) ** 2)
        loss /= self.mask_sum
        return loss, image_rgb

        # interception = torch.sum(torch.abs(image * self.image_ref[None, :, :]))
        # union = torch.sum(image) + torch.sum(self.image_ref) - interception
        # loss = - interception / union
        # return loss, image_rgb


def make_gif(filename, dir_tmp, remove_png=False):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob(os.path.join(dir_tmp, '_tmp_*.png'))):
            writer.append_data(imageio.imread(filename))
            if remove_png:
                os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()

    images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)


def get_updated_RT(vertices,
                   faces,
                   Rotation_Matrix,
                   T,
                   euler_angle,
                   mask_full_size,
                   masked_grayscale_img,
                   camera_matrix,
                   image_size,
                   loss_RT=0.1,  # Greyscale difference
                   num_epochs=50,
                   draw_flag=False,
                   output_gif=None,
                   lr=0.05,
                   fix_rot=False,
                   fix_trans=False,
                   fix_light_source=True):
    model = Model(vertices,
                  faces,
                  Rotation_Matrix,
                  T,
                  euler_angle,
                  mask_full_size,
                  masked_grayscale_img,
                  camera_matrix,
                  image_size=image_size,
                  loss_thresh=loss_RT,
                  fix_rot=fix_rot,
                  fix_trans=fix_trans,
                  fix_light_source=fix_light_source
                  )
    if draw_flag:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # optimizer_trans = torch.optim.Adam(model.renderer.t, lr=lr)
    # optimizer_eular_angle = torch.optim.Adam(model.renderer.R, lr=lr*lr_angle_ratio)

    if draw_flag:  # We don't save the images
        if not os.path.isdir(output_gif[:-4]):
            os.mkdir(output_gif[:-4])
        else:
            # remove old files and create new empty dir
            shutil.rmtree(output_gif[:-4])
            os.mkdir(output_gif[:-4])

    # We only keep the best max IoU Result
    min_loss = 1.0

    for i in range(num_epochs):
        optimizer.zero_grad()
        loss, image = model()
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            best_translation = model.renderer.t.detach().cpu().numpy()[0]
            best_rot_matrix = model.renderer.R.detach().cpu().numpy()[0]
            min_loss = loss.item()

        if draw_flag:  # We don't save the images
            image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            image = image / image.max()
            image[:, :, 1] += model.image_ref.detach().cpu().numpy()[0] * 0.5
            imsave(os.path.join(output_gif[:-4], '_tmp_%04d.png' % i), image)
            ### we print some updates
            print('Optimizing (loss %.4f)' % loss.data)
            updated_translation = model.renderer.t.detach().cpu().numpy()[0]
            original_translation = model.translation_original
            changed_dis = TranslationDistance(original_translation, updated_translation, abs_dist=False)
            print('Origin translation: %s - > updated tranlsation: %s. Changed distance: %.4f' % (
                np.array2string(np.array(original_translation)), np.array2string(updated_translation), changed_dis))
            if not fix_rot:
                rot_matrix = model.renderer.R.detach().cpu().numpy()[0]
                updated_euler_angle = rot2eul(rot_matrix, model.euler_original)
                changed_rot = RotationDistance(model.euler_original, updated_euler_angle)
                print('Origin eular angle: %s - > updated eular angle: %s. Changed rot: %.4f'
                      % (np.array2string(np.array(model.euler_original)), np.array2string(updated_euler_angle),
                         changed_rot))

        if loss.item() < model.loss_thresh:
            break

    if draw_flag:
        make_gif(output_gif, output_gif[:-4])

    return best_translation, rot2eul(best_rot_matrix, model.euler_original)


def finetune_RT(output,
                dataset,
                loss_grayscale_light=0.05,
                loss_grayscale_RT=0.05,
                loss_IoU=0.9,
                num_epochs=50,
                draw_flag=True,
                lr=0.05,  # lr=0.05,
                conf_thresh=0.8,
                tmp_save_dir='/data/Kaggle/wudi_data/tmp_output/',
                fix_rot=True,
                num_car_for_light_rendering=2):
    """
    We first get the lighting parameters: using 2 cars gray scale,
    then use grayscale loss and IoU loss to update T, and R(optional)
    :param outputs:
    :param dataset:
    :param loss_grayscale_light:
    :param loss_grayscale_RT: default: 0.05 is a good guess
    :param loss_IoU:
    :param num_epochs:  num epochs for both lighting and R,T
    :param draw_flag:
    :param lr:
    :param conf_thresh: confidence threshold for NMR process from bboxes, if lower, we will not process
                        this individual car--> because we don't care and accelerate the learning process
    :param tmp_save_dir: tmp saving directory for plotting .gif images
    :param fix_rot: fix rotation, if set to True, we will not learn rotation
    :param fix_trans:  fix translation, if set to True, we will not learn translation--> most likely we are
                        learning the lighting is set to True
    :param fix_light_source:  fix light source parameters if set to True
    :param num_car_for_light_rendering: default is 2 (consume 9 Gb GPU memory),
                                        for P100, we could use 3.
                                        We use the closest (smallest z) for rendering
                                        because the closer, the bigger car and more grayscale information.
    :return: the modified outputs
    """
    CAR_IDX = 2
    output_gif = None
    outputs_update = [output].copy()
    camera_matrix = dataset.camera_matrix.copy()
    camera_matrix[1, 2] -= 1480  # Because we have only bottom half
    # First we collect all the car instances info. in an image
    bboxes, segms, six_dof = output[0], output[1], output[2]
    car_cls_score_pred = six_dof['car_cls_score_pred']
    quaternion_pred = six_dof['quaternion_pred']
    trans_pred_world = six_dof['trans_pred_world']
    car_labels = np.argmax(car_cls_score_pred, axis=1)
    kaggle_car_labels = [dataset.unique_car_mode[x] for x in car_labels]
    car_names = [car_id2name[x].name for x in kaggle_car_labels]
    euler_angles = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])

    conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox
    conf_list = conf > conf_thresh
    # We choose the closest z two cars
    idx_conf = np.array([False] * len(conf))  # We choose only one car

    lighting_count = 0
    for close_idx in np.argsort(trans_pred_world[:, -1]):
        if conf_list[close_idx]:
            idx_conf[close_idx] = True
            lighting_count += 1
            if lighting_count >= num_car_for_light_rendering:
                break

    # Di Wu parrallise the code as below for one image per GPU
    rgb_image = imread(output[2]['file_name'])
    # convert the rgb image to grayscale
    grayscale_image = color.rgb2gray(rgb_image)

    vertices_img = []
    max_vertices = 0
    faces_img = []
    # there are in total 4999-5000 faces... we choose 4999 faces, for some car, not rendering one
    # face should be alright.
    min_faces = 4999
    Rotation_Matrix_img = []
    T_img = []
    euler_angles_img = []
    mask_img = []

    for car_idx in range(len(quaternion_pred)):
        # The the HTC predicted Mask which is served as the GT Mask
        segms_car = segms[CAR_IDX][car_idx]
        mask = maskUtils.decode(segms_car)
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

        vertices_img.append(vertices)
        max_vertices = max(vertices.shape[0], max_vertices)
        faces_img.append(faces)
        min_faces = min(faces.shape[0], min_faces)
        Rotation_Matrix_img.append(Rotation_Matrix)
        T_img.append(T)
        euler_angles_img.append(np.array([yaw, pitch, roll]))
        mask_img.append(mask)

    Rotation_Matrix_img = np.stack(Rotation_Matrix_img)
    T_img = np.stack(T_img)
    euler_angles_img = np.stack(euler_angles_img)
    mask_img = np.stack(mask_img)
    masked_grayscale_img = mask_img[idx_conf].sum(axis=0) * grayscale_image[1480:, :]
    masked_grayscale_img = masked_grayscale_img / masked_grayscale_img.max()
    # For vertices and faces each car will generate different
    vertices_img_all = np.zeros((len(vertices_img), max_vertices, 3))
    faces_img_all = np.zeros((len(faces_img), min_faces, 3))

    for i in range(len(vertices_img)):
        vertices_img_all[i, :vertices_img[i].shape[0], :] = vertices_img[i]
        faces_img_all[i, :, :] = faces_img[i][:min_faces, :]

    if draw_flag:
        output_gif = tmp_save_dir + '/' + output[2]['file_name'].split('/')[-1][:-4] + '.gif'

    # Lighting fine-tuning
    light_intensity_directional, light_intensity_ambient, light_direction = get_updated_lighting(
        vertices=vertices_img_all[idx_conf],
        faces=faces_img_all[idx_conf],
        Rotation_Matrix=Rotation_Matrix_img[idx_conf],
        T=T_img[idx_conf],
        euler_angle=euler_angles_img[idx_conf],
        mask_full_size=mask_img[idx_conf],
        masked_grayscale_img=masked_grayscale_img,
        camera_matrix=camera_matrix,
        image_size=(3384, 2710 - 1480),
        loss_thresh=loss_grayscale_light,
        num_epochs=num_epochs,
        draw_flag=draw_flag,
        output_gif=output_gif,
        lr=lr,
        fix_rot=True,
        fix_trans=True,
        fix_light_source=False)

    # Now we start to fine tune R, T
    for i, true_flag in enumerate(conf_list):
        if true_flag:
            if draw_flag:
                output_gif = tmp_save_dir + '/' + output[2]['file_name'].split('/')[-1][:-4] + '_' + str(i) + '.gif'
            # Now we consider only one masked grayscale car
            masked_grayscale_car = mask_img[i] * grayscale_image[1480:, :]
            # masked_grayscale_car = masked_grayscale_car / masked_grayscale_car.max()
            T_update, ea_update = get_updated_RT(vertices=vertices_img_all[None, i],
                                                 faces=faces_img_all[None, i],
                                                 Rotation_Matrix=Rotation_Matrix_img[None, i],
                                                 T=T_img[None, i],
                                                 euler_angle=euler_angles_img[i],
                                                 mask_full_size=mask_img[None, i],
                                                 masked_grayscale_img=masked_grayscale_car,
                                                 camera_matrix=camera_matrix,
                                                 image_size=(3384, 2710 - 1480),
                                                 loss_RT=loss_grayscale_RT,
                                                 num_epochs=num_epochs,
                                                 draw_flag=draw_flag,
                                                 output_gif=output_gif,
                                                 lr=lr,
                                                 fix_rot=fix_rot,
                                                 light_intensity_directional=light_intensity_directional,
                                                 light_intensity_ambient=light_intensity_ambient,
                                                 light_direction=light_direction
                                                 )

            if fix_rot:
                # we don't change the euler angle here
                R_update = -euler_angles_img[i][1], -euler_angles_img[i][0], -euler_angles_img[i][2]
            else:
                # We need to reverse here
                R_update = -ea_update[1], -ea_update[0], -ea_update[2]

            # outputs_update is a list of length 0
            outputs_update[0][2]['trans_pred_world'][i] = T_update
            euler_angles[i] = R_update

        if not fix_rot:
            outputs_update[0][2]['euler_angle'] = euler_angles

        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        output_name = tmp_save_dir + '/' + output[2]['file_name'].split('/')[-1][:-4] + '.pkl'
        mmcv.dump(outputs_update[0], output_name)
    return

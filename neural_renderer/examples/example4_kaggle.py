"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import pycocotools.mask as maskUtils

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
from nr_kaggle_utils import load_json_car_model, make_gif

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle, euler_to_Rot, rotation_matrix_to_euler_angles

from mmdet.datasets.car_models import car_id2name


class Model(nn.Module):
    def __init__(self, outputs, img_idx, car_idx,
                 difference_ratio=0.05):
        super(Model, self).__init__()
        # load .obj
        self.unique_car_mode = [2, 6, 7, 8, 9, 12, 14, 16, 18,
                                19, 20, 23, 25, 27, 28, 31, 32,
                                35, 37, 40, 43, 46, 47, 48, 50,
                                51, 54, 56, 60, 61, 66, 70, 71, 76]
        CAR_CLS_COCO = 2

        output = outputs[img_idx]
        bboxes, segms, six_dof = output[0], output[1], output[2]
        car_cls_score_pred = six_dof['car_cls_score_pred']
        quaternion_pred = six_dof['quaternion_pred']
        trans_pred_world = six_dof['trans_pred_world']
        car_labels = np.argmax(car_cls_score_pred, axis=1)
        kaggle_car_labels = [self.unique_car_mode[x] for x in car_labels]
        car_names = [car_id2name[x].name for x in kaggle_car_labels]

        # get rotation initialisation for a indiviual car
        euler_angle = np.array([quaternion_to_euler_angle(x) for x in quaternion_pred])
        ea = euler_angle[car_idx]
        yaw, pitch, roll = ea[0], ea[1], ea[2]
        yaw, pitch, roll = -pitch, -yaw, -roll
        R = euler_to_Rot(yaw, pitch, roll).T
        T = trans_pred_world[car_idx]
        # we save the original T and R here
        self.translation_original = T.copy()
        self.euler_original = np.array([yaw, pitch, roll])

        json_file = car_names[car_idx] + '.json'
        vertices, faces = load_json_car_model(json_file)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image, this will be the mask image (which we think is very accurate)
        segms_car = segms[CAR_CLS_COCO][car_idx]
        mask = maskUtils.decode(segms_car)
        mask_full_size = np.zeros((2710, 3384))
        mask_full_size[1480:, :] = mask
        # we set the loss threshold to stop perturbation
        self.mask_full_size = mask_full_size
        self.mask_sum = self.mask_full_size.sum()
        self.loss_thresh = difference_ratio * self.mask_sum
        if False:
            from mmcv import imwrite
            imwrite(mask_full_size*255, '/home/wudi/code/Kaggle_PKU_Baidu/neural_renderer/examples/data/mask_full_size.png')
        image_ref = torch.from_numpy(mask_full_size.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters

        camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                  [0, 2305.8757, 1354.9849],
                                  [0, 0, 1]], dtype=np.float32)
        self.register_buffer('K', torch.from_numpy(camera_matrix))
        image_size = (3384, 2710)
        # setup renderer
        renderer = nr.Renderer(image_size=image_size,
                               orig_size=image_size,
                               camera_mode='projection',
                               K=camera_matrix[None, :, :])
        renderer.R = nn.Parameter(torch.from_numpy(np.array(R[None, :], dtype=np.float32)))
        renderer.t = nn.Parameter(torch.from_numpy(np.array(T[None, :], dtype=np.float32)))
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        if False:
            from mmcv import imwrite
            image_first = image.detach().cpu().numpy()[0]
            image_overlap = self.mask_full_size + image_first
            imwrite(image_overlap*255/2, '/home/wudi/code/Kaggle_PKU_Baidu/neural_renderer/examples/data/image_overlap.png')
        #loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)

        interception = torch.sum(torch.abs(image * self.image_ref[None, :, :]))
        union = torch.sum(image) + torch.sum(self.image_ref) - interception
        loss = - interception /union
        return loss


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()

    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--valid_pred_file', type=str, default=os.path.join(
        '/data/Kaggle/cwx_data/all_yihao069e100s5070_resume92Dec24-08-50-226141a3d1_valid_ep99.pkl'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example4_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result_kaggle_'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    outputs = pkl.load(open(args.valid_pred_file, "rb"))
    # suppose we only evaluate the first car in the first image
    img_show = 'ID_0aa8f8389.jpg'
    for i in range(len(outputs)):
        if outputs[i][2]['file_name'].split('/')[-1] == img_show:
            img_idx = i
            break
    car_idx = 5
    output_gif = args.filename_output + img_show[:-4] + '_' + str(car_idx) + '.gif'

    model = Model(outputs, img_idx, car_idx)
    model.cuda()

    draw_flag = True
    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loop = tqdm.tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        if draw_flag:  # We don't save the images
            images = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
            image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
            imsave('/tmp/_tmp_%04d.png' % i, image)
        #loop.set_description('Optimizing (loss %.4f), %.3f NOT overlapping.' % (loss.data, loss.data/model.mask_sum))
        loop.set_description('Optimizing (loss %.4f)' % (loss.data))

        ### we print some updates
        if False:
            updated_translation = model.renderer.t.detach().cpu().numpy()
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
        #if loss.item() < model.loss_thresh:
        if loss.item() < -0.9:   # If IoU> 0.9
            break
    if draw_flag:
        make_gif(output_gif)


if __name__ == '__main__':
    main()

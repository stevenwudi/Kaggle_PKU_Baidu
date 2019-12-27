"""
Example 1. Drawing a Kaggle car from multiple vies.
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio
import json

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def load_car_models(car_model_dir='/data/Kaggle/pku-autonomous-driving/car_models_json'):
    car_model_dict = {}
    for car_name in tqdm(os.listdir(car_model_dir)[0]):
        with open(os.path.join(car_model_dir, car_name)) as json_file:
            car_model_dict[car_name[:-5]] = json.load(json_file)

    return car_model_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'kaggle_1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 10
    elevation = 10
    texture_size = 2

    # load .obj
    model_json = '/data/Kaggle/pku-autonomous-driving/car_models_json/biyadi-tang.json'
    with open(model_json) as json_file:
        car_model = json.load(json_file)

    vertices = np.array(car_model['vertices'])
    vertices[:, 1] = -vertices[:, 1]
    faces = np.array(car_model['faces']) - 1

    # to gpu
    vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
    faces = torch.from_numpy(faces.astype(np.int32)).cuda()

    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255 * image).astype(np.uint8))
    writer.close()


if __name__ == '__main__':
    main()

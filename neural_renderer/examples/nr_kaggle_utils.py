import json
import numpy as np
import torch
import imageio
import os
import glob


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def load_json_car_model(model_json, car_dir='/data/Kaggle/pku-autonomous-driving/car_models_json'):
    """
    Load kaggle car json object
    :param model_json:
    :return:
    """
    with open(os.path.join(car_dir, model_json)) as json_file:
        car_model = json.load(json_file)

    vertices = np.array(car_model['vertices'])
    vertices[:, 1] = -vertices[:, 1]
    faces = np.array(car_model['faces']) - 1

    # to gpu
    vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
    faces = torch.from_numpy(faces.astype(np.int32)).cuda()

    return vertices, faces

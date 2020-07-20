import os
from tqdm import tqdm
import json

car_model_dir = '/data/Kaggle/pku-autonomous-driving/car_models_json'
obj_output_dir = '/data/Kaggle/pku-autonomous-driving/car_model_obj'
car_model_dict = {}
for car_name in tqdm(os.listdir(car_model_dir)):
    with open(os.path.join(car_model_dir, car_name)) as json_file:
        json_dict = json.load(json_file)

    output_obj = os.path.join(obj_output_dir, car_name.replace('json', 'obj'))

    with open(output_obj, 'w') as f:
        f.write("# OBJ file\n")
        for vertices in json_dict['vertices']:
            f.write("v")
            for v in vertices:
                f.write(" %.4f" % v)
            f.write('\n')
        for faces in json_dict['faces']:
            f.write("f")
            for face in faces:
                f.write(" %d" % (face + 1))
            f.write("\n")
import os
from tqdm import tqdm
import json
import open3d as o3d
import numpy as np


def load_car_models(car_model_dir):
    car_model_dict = {}
    for car_name in tqdm(os.listdir(car_model_dir)):
        with open(os.path.join(car_model_dir, car_name)) as json_file:
            car_model_dict[car_name[:-5]] = json.load(json_file)

    return car_model_dict


def convert_to_obj(car_model_dir, obj_output_dir):
    car_models = load_car_models(car_model_dir)

    for car_name in car_models.keys():

        car_model = car_models[car_name]
        vertices = np.array(car_model['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        faces = np.array(car_model['faces'])

        # calculate norm
        mesh_car = o3d.geometry.TriangleMesh()
        mesh_car.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_car.triangles = o3d.utility.Vector3iVector(faces-1)
        # Computing normal
        mesh_car.compute_vertex_normals()
        # Get the vertice norm
        vertex_normals = np.asarray(mesh_car.vertex_normals)
        output_obj = os.path.join(obj_output_dir, car_name + '.obj')

        with open(output_obj, 'w') as f:
            f.write("# OBJ file\n")
            for idx, vertice in enumerate(vertices):
                f.write("v")
                for v in vertice:
                    f.write(" %.4f" % v)
                f.write('\n')
                # write vertex normals
                f.write("vn")
                for vn in vertex_normals[idx]:
                    f.write(" %.4f" % vn)
                f.write('\n')
            # write vertex texture coordinate according to number of faces
            for idx, face in enumerate(faces):
                f.write("vt")
                f.write(" %.4f %.4f" % (idx/len(faces), idx/len(faces)))
                f.write('\n')
            for idx, face in enumerate(faces):
                f.write("f")
                for face_idx, face_idv in enumerate(face):
                    write_idx = face_idx + idx
                    write_idx = min(max(write_idx, 1), len(faces)-1)
                    f.write(" %d/%d/%d" % (face_idv, write_idx, face_idv))
                f.write("\n")


if __name__ == '__main__':
    car_model_dir = 'E:\DATASET\pku-autonomous-driving\car_models_json'
    obj_output_dir = r'E:\CarInsurance\car_models_obj'
    convert_to_obj(car_model_dir, obj_output_dir)

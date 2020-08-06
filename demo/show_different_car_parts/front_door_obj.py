import json
import numpy as np
from demo.show_different_car_parts.convert_json_to_obj import convert_to_obj

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))


def obj_3d_generation(x, y, z, faces, vertices,
                      left_front_max_x, left_front_min_x, left_front_max_y, left_front_min_y,
                      left_front_max_z, left_front_min_z):
    points3D = np.vstack((x, y, z)).T
    tri_vertices = [points3D[index] for index in faces]

    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]
    ymean = [np.mean(tri[:, 1]) for tri in tri_vertices]
    xmean = [np.mean(tri[:, 0]) for tri in tri_vertices]

    I, J, K = tri_indices(faces)
    I_new = []
    J_new = []
    K_new = []

    vertices_new = []
    vertices_idx_new = []
    vertices_dict = {}
    vertices_count = 0
    faces_new = []
    for i, tri_i in enumerate(points3D[I]):
        if xmean[i] > left_front_min_x and xmean[i] < left_front_max_x:
            if ymean[i] > left_front_min_y and ymean[i] < left_front_max_y:
                if zmean[i] > left_front_min_z and zmean[i] < left_front_max_z:
                    I_new.append(I[i])
                    J_new.append(J[i])
                    K_new.append(K[i])

                    # We append the vertices here:
                    if I[i] not in vertices_idx_new:
                        vertices_idx_new.append(I[i])
                        vertices_new.append(vertices[I[i]])
                        vertices_dict[I[i]] = vertices_count
                        vertices_count += 1
                    if J[i] not in vertices_idx_new:
                        vertices_idx_new.append(J[i])
                        vertices_new.append(vertices[J[i]])
                        vertices_dict[J[i]] = vertices_count
                        vertices_count += 1
                    if K[i] not in vertices_idx_new:
                        vertices_idx_new.append(K[i])
                        vertices_new.append(vertices[K[i]])
                        vertices_dict[K[i]] = vertices_count
                        vertices_count += 1

                    # We save the face here

                    faces_new.append([vertices_dict[I[i]],
                                      vertices_dict[J[i]],
                                      vertices_dict[K[i]]])

    # We convert it back to the original json format
    vertices_gen = np.array(vertices_new)
    vertices_gen[: 1] *= 1
    faces_gen = np.array(faces_new) + 1

    out_obj_name = r'E:\CarInsurance\Car_15_seg_obj\aodi-Q7-SUV\aodi-Q7-SUV_font_door.obj'
    convert_to_obj(vertices_gen, faces_gen, out_obj_name)


with open(r'E:\CarInsurance\car_models_json_wd/aodi-Q7-SUV.json') as json_file:
    data = json.load(json_file)
    vertices, faces = np.array(data['vertices']), np.array(data['faces']) - 1

    x, y, z = vertices[:, 0], vertices[:, 2], -vertices[:, 1]
    car_type = data['car_type']

    # left_front_max_x = 1.5
    # left_front_min_x = 0.8

    left_front_min_x = -1.5
    left_front_max_x = -0.8

    left_front_max_y = 1.0
    left_front_min_y = -0.15

    left_front_max_z = 0.30
    left_front_min_z = -1.0

    graph_data = obj_3d_generation(x, y, z, faces, vertices,
                                   left_front_max_x, left_front_min_x, left_front_max_y,
                                   left_front_min_y, left_front_max_z, left_front_min_z)


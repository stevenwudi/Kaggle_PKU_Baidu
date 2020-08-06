import open3d as o3d
import numpy as np


def convert_to_obj(vertices, faces, out_obj_name):
    vertices[:, 1] = -vertices[:, 1]
    # calculate norm
    mesh_car = o3d.geometry.TriangleMesh()
    mesh_car.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_car.triangles = o3d.utility.Vector3iVector(faces - 1)
    # Computing normal
    mesh_car.compute_vertex_normals()
    # Get the vertice norm
    vertex_normals = np.asarray(mesh_car.vertex_normals)

    with open(out_obj_name, 'w') as f:
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
            f.write(" %.4f %.4f" % (idx / len(faces), idx / len(faces)))
            f.write('\n')
        for idx, face in enumerate(faces):
            f.write("f")
            for face_idx, face_idv in enumerate(face):
                write_idx = face_idx + idx
                write_idx = min(max(write_idx, 1), len(faces) - 1)
                f.write(" %d/%d/%d" % (face_idv, write_idx, face_idv))
            f.write("\n")

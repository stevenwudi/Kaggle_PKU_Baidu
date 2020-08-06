import matplotlib.cm as cm
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as FF
import plotly.graph_objs as go
import numpy as np


def tri_indices(simplices):
    # print('len(simplices)',len(simplices))
    # print('after tri',len(([triplet[c] for triplet in simplices] for c in range(3))))
    return ([triplet[c] for triplet in simplices] for c in range(3))


def plotly_trisurf(x, y, z, simplices, left_front_max_x, left_front_min_x, left_front_max_y, left_front_min_y,
                   left_front_max_z, left_front_min_z, colormap=cm.RdBu, plot_edges=None):
    points3D = np.vstack((x, y, z)).T
    tri_vertices = [points3D[index] for index in simplices]

    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]
    ymean = [np.mean(tri[:, 1]) for tri in tri_vertices]
    xmean = [np.mean(tri[:, 0]) for tri in tri_vertices]

    zmean_new = [np.mean(tri[:, 2]) for tri in tri_vertices]
    min_zmean = np.min(zmean)  # 按z轴将点排序
    max_zmean = np.max(zmean)

    min_ymean = np.min(ymean)  # 按z轴将点排序
    max_ymean = np.max(ymean)

    min_xmean = np.min(xmean)  # 按z轴将点排序
    max_xmean = np.max(xmean)

    facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]

    I, J, K = tri_indices(simplices)
    I_pt = points3D[I]
    J_pt = points3D[J]
    K_pt = points3D[K]

    I_new = []
    J_new = []
    K_new = []

    left_front_I = []
    left_front_J = []
    left_front_K = []

    for i, tri_i in enumerate(I_pt):
        if xmean[i] > left_front_min_x and xmean[i] < left_front_max_x:
            if ymean[i] > left_front_min_y and ymean[i] < left_front_max_y:
                if zmean[i] > left_front_min_z and zmean[i] < left_front_max_z:
                    I_new.append(I[i])
                    J_new.append(J[i])
                    K_new.append(K[i])

    triangles = go.Mesh3d(x=x, y=y, z=z,
                          facecolor=facecolor,
                          i=I_new, j=J_new, k=K_new,
                          name='')

    if plot_edges is None:
        return [triangles]
    else:
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

        lines = go.Scatter3d(x=Xe, y=Ye, z=Ze,
                             mode='lines',
                             line=dict(color='rgb(50,50,50)', width=1.5))
        return [triangles, lines]


def map_z2color(zval, colormap, vmin, vmax):
    if vmin > vmax: raise ValueError('incorrect relation between vmin and vmax')
    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    R, G, B, alpha = colormap(t)
    return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'.format(int(G * 255 + 0.5)) + \
           ',' + '{:d}'.format(int(B * 255 + 0.5)) + ')'


with open(r'E:\CarInsurance\car_models_json_wd/aodi-Q7-SUV.json') as json_file:
    data = json.load(json_file)
    vertices, triangles = np.array(data['vertices']), np.array(data['faces']) - 1

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

    graph_data = plotly_trisurf(x, y, z, triangles, left_front_max_x, left_front_min_x, left_front_max_y,
                                left_front_min_y, left_front_max_z, left_front_min_z, colormap=cm.RdBu, plot_edges=None)

    # # with no axis
    # noaxis=dict(showbackground=False,
    #         showline=False,
    #         zeroline=False,
    #         showgrid=False,
    #         showticklabels=False,
    #         title='')
    #
    # with axis
    axis = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
    )

    #     layout = go.Layout(
    #          title=car_type + ' with noaxis',
    #          width=800, height=600,
    #          scene=dict(
    #              xaxis= (noaxis), yaxis=dict(noaxis), zaxis=dict(noaxis),
    # #              aspectratio=dict( x=1, y=2, z=0.5)
    #          )
    #     )
    #
    #     fig = go.Figure(data= graph_data, layout=layout)
    #

    # fig.show()

    layout = go.Layout(
        title=car_type + ' with axis',
        width=800, height=600,
        scene=dict(
            xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis),
            #              aspectratio=dict( x=1, y=2, z=0.5)
        )
    )

    fig = go.Figure(data=graph_data, layout=layout)
    fig.update_layout(scene_aspectmode="data")

    fig.show()

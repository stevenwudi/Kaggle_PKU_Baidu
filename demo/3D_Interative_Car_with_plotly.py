import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as FF
import functools
import plotly.graph_objs as go
import matplotlib.cm as cm

PATH = 'E:\DATASET\pku-autonomous-driving'
with open(os.path.join(PATH, 'car_models_json/benchi-ML500.json')) as json_file:
    data = json.load(json_file)


def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))


def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    points3D = np.vstack((x, y, z)).T
    tri_vertices = map(lambda index: points3D[index], simplices)
    zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]
    min_zmean = np.min(zmean)
    max_zmean = np.max(zmean)
    facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
    I, J, K = tri_indices(simplices)

    triangles = go.Mesh3d(x=x, y=y, z=z,
                          facecolor=facecolor,
                          i=I, j=J, k=K,
                          name='')

    if plot_edges is None:
        return [triangles]
    else:
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

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


vertices, triangles = np.array(data['vertices']), np.array(data['faces']) - 1

x, y, z = vertices[:, 0], vertices[:, 2], -vertices[:, 1]
car_type = data['car_type']
graph_data = plotly_trisurf(x, y, z, triangles, colormap=cm.RdBu, plot_edges=None)

# with no axis
noaxis = dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='')

# with axis
axis = dict(
    showbackground=True,
    backgroundcolor="rgb(230, 230,230)",
    gridcolor="rgb(255, 255, 255)",
    zerolinecolor="rgb(255, 255, 255)",
)

layout = go.Layout(
    title=car_type + ' with noaxis',
    width=800, height=600,
    scene=dict(
        xaxis=dict(noaxis), yaxis=dict(noaxis), zaxis=dict(noaxis),
        #              aspectratio=dict( x=1, y=2, z=0.5)
    )
)

fig = go.Figure(data=graph_data, layout=layout)

fig.show()

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
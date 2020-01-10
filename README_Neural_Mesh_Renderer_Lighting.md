# Neural Mesh Renderer for lighting

## Why we need to learn lighting?

Because if only use silhouette to learn Rotation (R) and Translation (T), 
we observe that the NMR will mostly will use Rotation to minimise the silhouette 
IoU loss. See the example below (green is the ground truth mask):

![](./neural_renderer/examples/data/ID_2e24dd0ea_2.gif)

Hence, we need grayscale information as a stronger supervision.
But since we don't have the color and texture information for the vehicles, we can only utilise the mesh information.
And the direction and intensity of the lighting is important.

### Lighting

Lighting can be applied directly to a mesh. In NMR, there are ambient light
<img src="https://render.githubusercontent.com/render/math?math=l^a">
be the  intensities of the ambient light and 
<img src="https://render.githubusercontent.com/render/math?math=l^d">
directional light, respectively, 
<img src="https://render.githubusercontent.com/render/math?math=n^d">
be a unit vector indicating the direction of the directional light, and
<img src="https://render.githubusercontent.com/render/math?math=n_j">
be the normal vector of a surface.
The modified color of a pixel 
<img src="https://render.githubusercontent.com/render/math?math=I^l_j">
<img src="https://render.githubusercontent.com/render/math?math=I^l_j = (l^a + (n^d \cdot n_j)l^d)I_j">
.

In the NMR formulation, gradients also flow into the intensities 
as well as the direction of the directional light.
 Therefore, light sources can also be included as an optimisation target.


### Running examples: learning ambient light intensity, directional light intensity and directional light 



![](./neural_renderer/examples/data/mask_full_size.png)![](./neural_renderer/examples/data/image_overlap.png)![](./neural_renderer/examples/data/example4_result_kaggle_ID_0aa8f8389_0.gif)


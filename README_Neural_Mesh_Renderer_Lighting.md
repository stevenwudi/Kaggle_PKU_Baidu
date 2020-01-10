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
and directional light. Let <img src="https://latex.codecogs.com/gif.latex?$l^a$ /> and 
<img src="https://latex.codecogs.com/gif.latex?$l^d$ /> be the  intensities of the ambient light and 
directional light, respectively, 
<img src="https://latex.codecogs.com/gif.latex?$n^d$ /> be a unit vector indicating the 
direction of the directional light, and $n_j$ be the normal vector of a surface.
The modified color of a pixel 
<img src="https://latex.codecogs.com/gif.latex?$I^l_j$ /> on the surface will be
<img src="https://latex.codecogs.com/gif.latex? I^l_j = (l^a + (n^d \cdot n_j)l^d)I_j$ />.

In the NMR formulation, gradients also flow into the intensities 
<img src="https://latex.codecogs.com/gif.latex?$l^a$ /> and 
<img src="https://latex.codecogs.com/gif.latex?$l_d$ />,
as well as the direction 
<img src="https://latex.codecogs.com/gif.latex?$n^d$ />of the directional light.
 Therefore, light sources can also be included as an optimisation target.


### Running examples: learning ambient light intensity, directional light intensity and directional light 



![](./neural_renderer/examples/data/mask_full_size.png)![](./neural_renderer/examples/data/image_overlap.png)![](./neural_renderer/examples/data/example4_result_kaggle_ID_0aa8f8389_0.gif)


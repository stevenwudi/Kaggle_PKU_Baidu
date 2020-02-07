## Instruction for training with camera rotation

We follow the camera rotation augmentation as in
https://www.kaggle.com/outrunner/rotation-augmentation
It's put both in the `kaggle_pku._parse_ann_info`
and the in the `dataset.pipeline.transform`.
The reason is that we need to generate mask and bounding box
for the camera rotation in each iteration:
```python
            alpha = ((np.random.random()**0.5)*8-5.65)*np.pi/180.
            beta = (np.random.random() * 50 - 25) * np.pi / 180.
            gamma = (np.random.random() * 6 - 3) * np.pi / 180. + beta / 3
```
 
 But the actually loading of the image is saved in the `dataset.pipeline.transform`.
 So we save the image rotation matrix `Mat` from`Mat, Rot = self.rotateImage(alpha, beta, gamma)`
 and pass it to  the `CameraRotation` class:
 
 `results['img'] = cv2.warpPerspective(img, trans, (w, h), flags=cv2.INTER_LANCZOS4)`
 
 
## What need to be added in the configs file

(1) Set the `rotation_augmenation` in `data.train` as True (default is False).

```python
data = dict(
    train=dict(
        rotation_augmenation=True),
        ...
        )
```
(2) add CameraRotation in train_pipeline
```python
train_pipeline = [
    dict(type='CameraRotation'),
    ...
    ]
```


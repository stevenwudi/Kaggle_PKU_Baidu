# Kaggle_PKU_Baidu

##  Neural Mesh Renderer

We follow the github repo [chainer](https://github.com/hiroharu-kato/neural_renderer), and there is
a [pytorch](https://github.com/daniilidis-group/neural_renderer) version.

But finally we used the [https://github.com/hzsydy/neural_renderer](https://github.com/hzsydy/neural_renderer)
because it allows image width and height to be different.

### how to install

go to the directory `neural_renderer` and install by
`python setup.py develop`

### Running examples: learning R and T 

`python ./neural_renderer/examples/example4_kaggle.py`

![](./neural_renderer/examples/data/mask_full_size.png)![](./neural_renderer/examples/data/image_overlap.png)![](./neural_renderer/examples/data/example4_result_kaggle_ID_0aa8f8389_0.gif)


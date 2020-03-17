# 2nd Place for Kaggle_PKU_Baidu Solution

Below you can find a outline of how to reproduce our solution for the
[Peking University/Baidu - Autonomous Driving](https://www.kaggle.com/c/pku-autonomous-driving/) competition.

The solution is detailed in [Solution](README_solution.md).

### ARCHIVE CONTENTS

One trained model can be found at [Google Drive](https://drive.google.com/open?id=1IldUtfgoRly6Ili3C9h6Xncgfet4DXKC).
It achieves 0.112/0.118 on (private/public Leaderboard).


## Installation
### Requirements


We have tested the following versions of OS and softwares:
- OS: Ubuntu 16.04 LTS 
- nvidia drivers v.384.130
- 4 x NVIDIA GeForce GTX 1080

- Python 3.6.9
- CUDA 9.0
- cuddn 7.4
- pytorch 1.1 (or +)
- GCC(G++): 4.9/5.3/5.4/7.3
- mmdet: 1.0.rc0+d3ca926  
(Or you can install the mmdet from the uploaded files. The newest mmdet 1.4+ has different API in calling mmcv.
Hence, we would recommend install the mmdet from the uploaded files using:
`python setup.py install`)

### Configurations
 All the data and network configurations are in the .config file:
 ` ./configs/htc/htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation.py ` 


#### Dataset setup

Download the dataset from Kaggle and modify the config file to where you download the data
`data_root = '/data/Kaggle/pku-autonomous-driving/'`

The newly added dataloader from `mmdet/datasets/kaggle_pku.py`
will generate a corresponding annotation json file.

### Running the code

#### training
In the tools folder, we provide the scripts for training:

- single gpu training: `python train_kaggle_pku.py `: currently single gpu training does not support validation evaluation. It
is recommended to  use the below

- multi-gpu training:  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 train_kaggle_pku.py --launcher pytorch`

- validation is only written for distributed training. If you only have a single gpu, you can do something like:  `CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_kaggle_pku.py --launcher pytorch`

It is also recommended to load some pretrained model from mmdet.

Note: `kaggle_apollo_combined_6691_origin.json` is the annotation file from the combination of ApolloScape and Kaggle (We also cleaned up the noisy images with mesh overlay visualisation).
Alternatively, you can use `train.json` from the Kaggle file. 
We have uploaded a json file `kaggle_apollo_combined_6691_origin.json` to google drive as:
（We have encoded an absolute path, so to use the clean up data, you need to change the filepath
according to your data location--> both for Kaggle and Apolloscape)

https://drive.google.com/open?id=1gEK7aGvTSAi8o2Jq3PgFky_YYX-HpJDt

#### Inference 

- single model inference:  the inference time is around 3 fps. `python test_kaggle_pku.py` and it will generate a single model .csv file.
post processing

#### Model merging

`python tools/model_merge.py`  in this script, you need to set the corresponding generated predicted pickle file.

## Cite

If you find this repo helpful, we would appreciate if you cite the following paper.

```
@article{NMR6D2020,
  title   = {Neural Mesh Refiner for 6-DoF Pose Estimation},
  author  = {Di Wu and Yihao Chen and Xianbiao Qi and Yuyong Jian and Weixuan Chen and Rong Xiao},
  journal= {arXiv preprint arXiv:3083918},
  year={2020}
}
```

## Contact

This repo is currently maintained by Di Wu ([@stevenwudi](http://github.com/stevenwudi)) and Yihao Chen ([@cyh1112](o0o@o0oo0o.cc)).


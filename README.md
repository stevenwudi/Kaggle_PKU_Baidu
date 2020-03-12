# 2nd Place for Kaggle_PKU_Baidu Solution

Below you can find a outline of how to reproduce my solution for the
[Peking University/Baidu - Autonomous Driving](https://www.kaggle.com/c/pku-autonomous-driving/) competition.

The solution is detailed in [Solution](README_solution.md).

### ARCHIVE CONTENTS

One trained model can be found at [Google Drive](https://drive.google.com/open?id=0BzicoAl6Jud9U1hMUEw0N1R4VlRHeUpwdll4VTBqMnAxT29z).


## Installation
### Requirements


We have tested the following versions of OS and softwares:
- OS: Ubuntu 16.04 LTS 
- nvidia drivers v.384.130
- 4 x NVIDIA GeForce GTX 1080

- Python 3.6.9
- CUDA 9.0
- cuddn 7.4
- GCC(G++): 4.9/5.3/5.4/7.3
- mmdet: 1.0.rc0+d3ca926  (the uploa)

### Configurations
 All the data and network configurations are in the .config file:
 ` ./configs/htc/htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi.py ` 


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

It is also recommended to load some pretrained model from mmdet.

#### Inference 

- single model inference:  the inference time is around 3 fps. `python test_kaggle_pku.py` and it will generate a single model .csv file.
post processing

#### Model merging

`python tools/model_merge.py`  in this script, you need to set the corresponding generated predicted pickle file.


## Contact

This repo is currently maintained by Di Wu ([@stevenwudi](http://github.com/stevenwudi)).
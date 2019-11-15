import os
from tqdm import tqdm
os.environ['CUDA_VISIABLE_DEVICES'] = '0'

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.coco import CocoDataset
from visualisation_utils import show_result_kaggle_pku

config_file = '../configs/htc/htc_hrnetv2p_w48_20e_kaggle_pku.py'
checkpoint_file ='/data/Kaggle/wudi_data/work_dirs/htc_hrnetv2p_w48_20e_kaggle_pku_Nov13-11-15-55/epoch_8.pth'
model = init_detector(config_file, checkpoint_file)

img_dir = '/data/Kaggle/pku-autonomous-driving/test_images'
out_dir = '/data/Kaggle/wudi_data/mask_test'
img_names = os.listdir(img_dir)


for i in tqdm(range(len(img_names))):

    img = os.path.join(img_dir, img_names[i])
    result = inference_detector(model, img)
    show_result_kaggle_pku(img, result,
                           CocoDataset.CLASSES,
                           show=False,
                           out_file=os.path.join(out_dir, img_names[i]))


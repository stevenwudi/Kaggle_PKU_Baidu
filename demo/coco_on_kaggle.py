import os
from tqdm import tqdm
os.environ['CUDA_VISIABLE_DEVICES'] = '0'

from mmdet.apis import init_detector, inference_detector, show_result


config_file = '../configs/hrnet/htc_hrnetv2p_w48_20e.py'
checkpoint_file = '/data/Kaggle/mmdet_pretrained_weights/htc_hrnetv2p_w48_28e_20190810-a4274b38.pth'

model = init_detector(config_file, checkpoint_file)

img_dir = '/data/Kaggle/pku-autonomous-driving/train_images'
out_dir = '/data/Kaggle/wudi_data/coco_train_image_vis'
img_names = os.listdir(img_dir)
for i in tqdm(range(len(img_names))):

    img = os.path.join(img_dir, img_names[i])
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, show=False,
                out_file=os.path.join(out_dir, img_names[i]))
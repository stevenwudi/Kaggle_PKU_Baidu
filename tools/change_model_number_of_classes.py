from collections import OrderedDict
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

in_file = '/data/Kaggle/wudi_data/work_dirs/htc_hrnetv2p_w48_20e_kaggle_pku_Nov13-11-15-55/epoch_2.pth'
out_file = '/data/Kaggle/mmdet_pretrained_weights/trimmed_htc_hrnetv2p_w48_20e_kaggle_pku.pth'


# in_file = '/data/Kaggle/mmdet_pretrained_weights/htc_hrnetv2p_w48_28e_20190810-a4274b38.pth'
# out_file = '/data/Kaggle/mmdet_pretrained_weights/trimmed_hrnetv2p_w48_28e_20190810-a4274b38.pth'
#

checkpoint = torch.load(in_file)
in_state_dict = checkpoint.pop('state_dict')
out_state_dict = OrderedDict()

delete_dict = ['semantic_head']

for key, val in in_state_dict.items():
    find_flag = False
    for delete_key in delete_dict:
        if delete_key in key:
            print('Deleteting: %s' % key)
            find_flag = True

    if not find_flag:
        out_state_dict[key] = val

checkpoint['state_dict'] = out_state_dict

if 'optimizer' in checkpoint:
    del checkpoint['optimizer']
torch.save(checkpoint, out_file)
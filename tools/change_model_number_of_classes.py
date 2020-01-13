from collections import OrderedDict
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

in_file = '/data/Kaggle/checkpoints/all_cwxe99_3070100flip05resumme93Dec29-16-28-48/epoch_100.pth'
out_file = '/data/Kaggle/checkpoints/all_cwxe99_3070100flip05resumme93Dec29-16-28-48_trimmed_translation.pth'

checkpoint = torch.load(in_file)
in_state_dict = checkpoint.pop('state_dict')
out_state_dict = OrderedDict()

delete_dict = ['translation_head']

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
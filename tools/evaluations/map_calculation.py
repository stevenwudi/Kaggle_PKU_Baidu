import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from math import acos, pi
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from multiprocessing import Pool
import os

from mmdet.utils import check_match, expand_df

def match(t):
    return check_match(*t)

if __name__ == '__main__':
    validation_prediction = '/data/Kaggle/wudi_data/work_dirs/validation_Dec01-10-14-39_validation_images_conf_0.1.csv'
    car_conf_score_thres = 0.1
    valid_df = pd.read_csv(validation_prediction)
    valid_df.ImageId = [x.replace('.jpg', '') for x in os.listdir('/data/Kaggle/pku-autonomous-driving/validation_images/')]
    valid_df = valid_df.fillna('')
    train_df = pd.read_csv('/data/Kaggle/pku-autonomous-driving/train.csv')
    train_df = train_df[train_df.ImageId.isin(valid_df.ImageId.unique())]

    # data description page says, The pose information is formatted as
    # model type, yaw, pitch, roll, x, y, z
    # but it doesn't, and it should be
    # model type, pitch, yaw, roll, x, y, z
    expanded_train_df = expand_df(train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])

    max_workers = 10
    n_gt = len(expanded_train_df)
    ap_list = []
    p = Pool(processes=max_workers)
    for result_flg, scores in p.imap(match, [(i, train_df, valid_df) for i in range(10)]):
        if np.sum(result_flg) > 0:
            n_tp = np.sum(result_flg)
            recall = n_tp / n_gt
            ap = average_precision_score(result_flg, scores) * recall
        else:
            ap = 0
        ap_list.append(ap)
    map = np.mean(ap_list)
    print('map:', map)
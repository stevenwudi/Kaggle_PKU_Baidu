import mmcv
import os
import sys
import numpy as np

def write_submission(outputs):
    import pandas as pd
    import numpy as np
    from scipy.special import softmax
    from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle
    submission = 'Nov20-18-24-45-epoch_50.csv'

    predictions = {}
    PATH = '/data/Kaggle/pku-autonomous-driving/'
    ImageId =[i.strip() for i in open(PATH + 'validation.txt').readlines()]
    # ImageId = [x.replace('.jpg', '') for x in os.listdir(PATH + 'test_images')]

    for idx, output in enumerate(outputs):
        conf = np.max(softmax(output[2]['car_cls_score_pred'], axis=1), axis=1)
        euler_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])
        translation = output[2]['trans_pred_world']
        coords = np.hstack((euler_angle, translation, conf[:, None]))
        coords_str = coords2str(coords)
        try:
            predictions[ImageId[idx]] = coords_str
        except:
            continue

    
    pred_dict = {'ImageId':[],'PredictionString':[]}
    for k,v in predictions.items():
        pred_dict['ImageId'].append(k)
        pred_dict['PredictionString'].append(v)

    df = pd.DataFrame(data=pred_dict)
    print('df',df)
    # test = pd.read_csv(PATH + 'sample_submission.csv')
    # for im_id in test['ImageId']:
    #     test.loc[test['ImageId'] == im_id, ['PredictionString']] = [predictions[im_id]]

    df.to_csv(submission, index=False)


def coords2str(coords):
    s = []
    for c in coords:
        for l in c:
            s.append('%.5f'%l)
    return ' '.join(s)

if __name__ == '__main__':
    outputs = mmcv.load('/data/Kaggle/wudi_data/work_dirs/Nov20-18-24-45-epoch_50.pkl')
    write_submission(outputs)

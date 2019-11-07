"""
https://www.kaggle.com/greatgamedota/cv-util-functions
"""
import numpy as np
import pandas as pd
from math import sqrt, acos, pi
from scipy.spatial.transform import Rotation as R


thres_rot = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
thres_pos = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]


def trans_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def rot_dist(true, pred):
    x, y, z = true
    true = [z, x, y]
    x, y, z = pred
    pred = [z, x, y]
    q1 = R.from_euler('zyx', true)
    q2 = R.from_euler('zyx', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    W = 180 - (acos(W) * 180) / pi
    return W


def get_acc(true, pred):
    pred_rot = pred[:-3]
    true_rot = true[:-3]
    pred_pos = pred[3:]
    true_pos = true[3:]

    rot_d = rot_dist(true_rot, pred_rot)
    tran_d = trans_dist(true_pos, pred_pos)
    abs_dist = sqrt(sum([x**2 for x in true_pos]))
    print(rot_d, tran_d)

    thres = []
    for t in thres_rot:
        if rot_d < t:
            thres.append(1)
        else:
            thres.append(0)
    for t in thres_pos:
        if tran_d/abs_dist < t:
            thres.append(1)
        else:
            thres.append(0)
    print(thres)
    true_thres = np.ones((20))

    return apk(true_thres, thres, k=20)


# Ref: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py'''


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # if p in actual and p not in predicted[:i]:
        if p == actual[i]:
            num_hits += 1.0
            #             score += 1 / (i+1.0)
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


'''Ref: https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric'''
def map_per_set(labels, predictions, k=1):
    scores = []
    for idx,l in enumerate(labels):
        try:
            scores.append(1 / (predictions[:k].index(l) + 1.))
        except ValueError:
            scores.append(0)
    return np.mean(scores)

pred = [0.15, -3.1, -3.1, 8.09433326, 5.27078698, 21.43466666]
true = [0.169198, -3.11164, -3.08562, 7.42949, 4.99111, 20.2823]

print(get_acc(true, pred))


import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import os.path as osp
import shutil
import tempfile
import pandas as pd
import numpy as np
import random
import glob 

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from mmdet.datasets.kaggle_pku_utils import quaternion_to_euler_angle, filter_igore_masked_using_RT, filter_output
from tqdm import tqdm
from tools.evaluations.map_calculation import map_main
from multiprocessing import Pool

from finetune_RT_NMR import finetune_RT


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def write_submission(outputs, args, dataset,
                     conf_thresh=0.1,
                     filter_mask=False,
                     horizontal_flip=False):
    img_prefix = dataset.img_prefix

    submission = args.out.replace('.pkl', '')
    submission += '_' + img_prefix.split('/')[-1]
    submission += '_conf_' + str(conf_thresh)

    if filter_mask:
        submission += '_filter_mask.csv'
    elif horizontal_flip:
        submission += '_horizontal_flip'
    submission += '_merged.csv'
    predictions = {}

    CAR_IDX = 2  # this is the coco car class
    for idx_img, output in tqdm(enumerate(outputs)):
        file_name = os.path.basename(output[2]["file_name"])
        ImageId = ".".join(file_name.split(".")[:-1])

        # Wudi change the conf to car prediction
        if len(output[0][CAR_IDX]):
            conf = output[0][CAR_IDX][:, -1]  # output [0] is the bbox
            idx_conf = conf > conf_thresh
            if filter_mask:
                # this filtering step will takes 2 second per iterations
                #idx_keep_mask = filter_igore_masked_images(ImageId[idx_img], output[1][CAR_IDX], img_prefix)
                idx_keep_mask = filter_igore_masked_using_RT(ImageId, output[2], img_prefix, dataset)

                # the final id should require both
                idx = idx_conf * idx_keep_mask
            else:
                idx = idx_conf
            #if 'euler_angle' in output[2].keys():
            if False:   #NMR has problem saving 'euler angle' Its
                eular_angle = output[2]['euler_angle']
            else:
                eular_angle = np.array([quaternion_to_euler_angle(x) for x in output[2]['quaternion_pred']])
            translation = output[2]['trans_pred_world']
            coords = np.hstack((eular_angle[idx], translation[idx], conf[idx, None]))

            coords_str = coords2str(coords)

            predictions[ImageId] = coords_str
        else:
            predictions[ImageId] = ""

    pred_dict = {'ImageId': [], 'PredictionString': []}
    for k, v in predictions.items():
        pred_dict['ImageId'].append(k)
        pred_dict['PredictionString'].append(v)

    df = pd.DataFrame(data=pred_dict)
    print("Writing submission csv file to: %s" % submission)
    df.to_csv(submission, index=False)
    return submission


def filter_output_pool(t):
    return filter_output(*t)


def write_submission_pool(outputs, args, dataset,
                     conf_thresh=0.1,
                     horizontal_flip=False,
                     max_workers=20):
    """
    For accelerating filter image
    :param outputs:
    :param args:
    :param dataset:
    :param conf_thresh:
    :param horizontal_flip:
    :param max_workers:
    :return:
    """
    img_prefix = dataset.img_prefix

    submission = args.out.replace('.pkl', '')
    submission += '_' + img_prefix.split('/')[-1]
    submission += '_conf_' + str(conf_thresh)
    submission += '_filter_mask.csv'
    if horizontal_flip:
        submission += '_horizontal_flip'
    submission += '.csv'
    predictions = {}

    p = Pool(processes=max_workers)
    for coords_str, ImageId in p.imap(filter_output_pool, [(i, outputs, conf_thresh, img_prefix, dataset) for i in range(len(outputs))]):
        predictions[ImageId] = coords_str

    pred_dict = {'ImageId': [], 'PredictionString': []}
    for k, v in predictions.items():
        pred_dict['ImageId'].append(k)
        pred_dict['PredictionString'].append(v)

    df = pd.DataFrame(data=pred_dict)
    print("Writing submission csv file to: %s" % submission)
    df.to_csv(submission, index=False)
    return submission


def coords2str(coords):
    s = []
    for c in coords:
        for l in c:
            s.append('%.5f' % l)
    return ' '.join(s)





def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config',
                        default='../configs/htc/htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi.py',
                        help='train config file path')
    # parser.add_argument('--checkpoint', default='/data/Kaggle/cwx_data/htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_adam_pre_apollo_30_60_80_Dec07-22-48-28/epoch_58.pth', help='checkpoint file')
    parser.add_argument('--checkpoint',
                        default='/nfsc/vcdata/cyh/epoch_147.pth',
                        help='checkpoint file')
    parser.add_argument('--conf', default=0.1, help='Confidence threshold for writing submission')
    parser.add_argument('--json_out', help='output result file name without extension', type=str)
    parser.add_argument('--eval', type=str, nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints', ' kaggle'],
                        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--local_rank', help='show results')
    parser.add_argument('--tmpdir', default="./results/", help='tmp dir for writing some results')
    parser.add_argument('--clear', default=False, help='tmp dir for writing some results')
    parser.add_argument('--horizontal_flip',  default=False, action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def output_sorted(outputs):
    j = len(outputs)

    new_outputs = [[] for _ in range(j)]
    for output in outputs[0]:
        file_name = output[2]['file_name']
        new_outputs[0].append(output)

        for i in range(1, j):
            for output_i in outputs[i]:
                if file_name == output_i[2]['file_name']:
                    new_outputs[i].append(output_i)
                    break

    return new_outputs

def create_lock(file_name):
    f = open(file_name, mode="w", encoding="utf-8")
    f.close()

def remove_lock(file_name):
    os.remove(file_name)

def main():
    args = parse_args()

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # Wudi change the args.out directly related to the model checkpoint file data
    if args.horizontal_flip:
        args.out = os.path.join(cfg.work_dir,
                                cfg.data.test.img_prefix.split('/')[-2].replace('_images', '_') +
                                args.checkpoint.split('/')[-1][:-4] + '_horizontal_flip.pkl')
    else:
        args.out = os.path.join(cfg.work_dir,
                                cfg.data.test.img_prefix.split('/')[-2].replace('_images', '_') +
                                args.checkpoint.split('/')[-1][:-4] + '.pkl')

        # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    outputs_1 = mmcv.load('/data/Kaggle/cwx_data/all_cwxe99_3070100flip05resumme72Dec27-09-40-17/all_cwxe99_3070100flip05resumme72Dec27-09-40-17ep73.pkl')
    outputs_2 = mmcv.load('/data/Kaggle/cwx_data/all_cwxe99_3070100flip05resumme72Dec27-09-40-17/all_cwxe99_3070100flip05resumme72Dec27-09-40-17ep85.pkl')
    outputs_3 = mmcv.load('/data/Kaggle/cwx_data/all_cwxe99_3070100flip05resumme72Dec27-09-40-17/all_cwxe99_3070100flip05resumme72Dec27-09-40-17ep85.pkl')
    
    random.shuffle(outputs_1)
    outputs = output_sorted([outputs_1, outputs_2, outputs_3])
    
    for i in range(len(outputs[0])):
        output = outputs[0][i]
        lock_file = os.path.join(args.tmpdir, os.path.basename(output[2]['file_name']).replace(".jpg", ".lock"))
        pkl_file = os.path.join(args.tmpdir, os.path.basename(output[2]['file_name']).replace(".jpg", ".pkl"))

        if os.path.exists(lock_file) or os.path.exists(pkl_file):
            continue

        create_lock(lock_file)
        print(pkl_file)
        dataset.distributed_visualise_pred_merge_postprocessing(i, outputs, args, tmp_dir=args.tmpdir, vote=2)
        remove_lock(lock_file)

    pkl_list = glob.glob(os.path.join(args.tmpdir, "*.pkl"))

    if len(pkl_list) == len(outputs[0]):
        outputs_merged = []
        for pkl in pkl_list:
            output = mmcv.load(pkl)
            outputs_merged.append(output)

        print("Writing pkl file to: {}".format(args.out))
        mmcv.dump(outputs_merged, args.out)
        
        submission = write_submission(outputs_merged, args, dataset,
                                      conf_thresh=0.8,
                                      filter_mask=False,
                                      horizontal_flip=args.horizontal_flip)
        

if __name__ == '__main__':
    main()

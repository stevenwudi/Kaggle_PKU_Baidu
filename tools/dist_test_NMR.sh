#!/bin/bash


for((i = 0; i < 3; i ++))
do
    CUDA_VISIBLE_DEVICES=0,6,7  python  test_kaggle_pku.py --local_rank=$i --world_size=3 &
done



CUDA_VISIBLE_DEVICES=0  python  test_kaggle_pku.py --local_rank=0 --world_size=3

CUDA_VISIBLE_DEVICES=6  python  test_kaggle_pku.py --local_rank=1 --world_size=3

CUDA_VISIBLE_DEVICES=7  python  test_kaggle_pku.py --local_rank=2 --world_size=3
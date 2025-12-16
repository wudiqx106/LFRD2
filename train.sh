#! /bin/bash

#CUDA_VISIBLE_DEVICES=1 python ./main.py --prop_time 6 --is_synthetic --train --load_ckt
#echo 'Complete training on synthetic dataset!'
#CUDA_VISIBLE_DEVICES=1 python ./main.py --prop_time 6 --load_ckt --train
#echo 'Complete training on real dataset!'
CUDA_VISIBLE_DEVICES=1 python ./main.py --prop_time 6 --load_ckt --is_synthetic
#CUDA_VISIBLE_DEVICES=1 python ./main.py --prop_time 6 --load_ckt

#echo 'Complete evaluation!'
echo 'Done!'

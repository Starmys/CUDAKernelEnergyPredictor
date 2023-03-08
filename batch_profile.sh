#!/bin/bash

GPUS=( 0 1 2)
ALGOS=(4 5 6)  # 0 1 2

for i in {0..2}
do
  nohup python batch_profile.py --gpu=${GPUS[i]} --algo=${ALGOS[i]} > tmp/profile-${GPUS[i]}.out &! 
done  

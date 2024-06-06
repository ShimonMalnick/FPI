#!/bin/bash

cd /storage/malnick/FPI/ && conda init bash
echo "source activate fpi" > ~/.bashrc && source ~/.bashrc
nohup python examine_likelihoods_partial_inverse.py > ../logs/out_coco.log 2> ../logs/err_coco.log
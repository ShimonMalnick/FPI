#!/bin/bash
#nohup python examine_likelihoods_partial_inverse.py 0 > out_0.log 2> err_0.log &
#nohup python examine_likelihoods_partial_inverse.py 1 > out_1.log 2> err_1.log &
#nohup python examine_likelihoods_partial_inverse.py 2 > out_2.log 2> err_2.log &
#nohup python examine_likelihoods_partial_inverse.py 3 > out_3.log 2> err_3.log &
cd /storage/malnick/FPI/ && conda init bash
echo "source activate fpi" > ~/.bashrc && source ~/.bashrc
nvidia-smi
nohup python examine_likelihoods_partial_inverse.py > out_noisy.log 2> err_noisy.log
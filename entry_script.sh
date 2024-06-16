#!/bin/bash

cd /storage/malnick/FPI/null_space && conda init bash
echo "source activate fpi" > ~/.bashrc && source ~/.bashrc
wandb login 7a46bfc8dff80b42bd3001321c9b895b663cc318
nohup python train.py --attention_trainable key --output_dir results_sd_likelihood_key > ../../logs/out_sd_likelihood_key.log 2> ../../logs/err_sd_likelihood_all.log
#!/bin/bash

cd /storage/malnick/FPI/null_space && conda init bash
echo "source activate fpi" > ~/.bashrc && source ~/.bashrc
wandb login 7a46bfc8dff80b42bd3001321c9b895b663cc318
nohup python train.py --attention_trainable value --output_dir results_null_space_values > ../../logs/out_values.log 2> ../../logs/err_values.log
#python train.py --attention_trainable query --output_dir results_null_space_queries
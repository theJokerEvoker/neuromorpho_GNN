#!/bin/bash
#SBATCH --job-name=dl_neurons_human
#SBATCH --output=out_dl_human
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
cd ~/SGMP_code-main/data/NeuroMorpho/human
python3 batch_swc_download.py --dataset human

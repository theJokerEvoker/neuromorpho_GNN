#!/bin/bash
#SBATCH --job-name=dl_neurons_rat
#SBATCH --output=out_dl_rat
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
cd ~/SGMP_code-main/data/NeuroMorpho/rat
python3 batch_swc_download.py --dataset rat

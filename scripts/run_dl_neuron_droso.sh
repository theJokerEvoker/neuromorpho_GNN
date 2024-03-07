#!/bin/bash
#SBATCH --job-name=dl_neurons_droso
#SBATCH --output=out_dl_droso
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
cd ~/SGMP_code-main/data/NeuroMorpho/drosophila
python3 batch_swc_download.py --dataset droso
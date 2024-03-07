#!/bin/bash
#SBATCH --job-name=dl_neurons_mouse
#SBATCH --output=out_dl_mouse
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
cd ~/SGMP_code-main/data/NeuroMorpho/mouse
python3 batch_swc_download.py --dataset mouse

#!/bin/bash
#SBATCH --job-name=count_mouse
#SBATCH --output=out_count_class_cell_mouse_3
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
cd ~/SGMP_code-main
python3 class_numbers.py --dataset human

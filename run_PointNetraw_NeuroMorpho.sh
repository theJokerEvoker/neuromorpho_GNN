#!/bin/bash
#SBATCH --job-name=PointNetraw_5xfad_glia
#SBATCH --output=out_PointNetraw_Neuro_5xfad_glia_200_1e4
#SBATCH --gres=gpu:1

cd ~/.
source ~/anaconda3/bin/activate
python3 ~/SGMP_code-main/main_base.py --save_dir ./results --data_dir ./data --model PointNetraw --dataset Neuro_exp_cond --split 811 --device gpu --random_seed 69 --batch_size 16 --epoch 200 --lr 1e-4 --test_per_round 5 --label 0 --num_layers 3

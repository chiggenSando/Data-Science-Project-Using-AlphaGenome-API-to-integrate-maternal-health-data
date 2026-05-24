#!/bin/bash
#SBATCH --job-name=alphagenome_finetune
#SBATCH --output=/opt/home/s4076520/alphagenome/logs/finetune_%j.txt
#SBATCH --error=/opt/home/s4076520/alphagenome/logs/finetune_%j.err
#SBATCH --partition=SCT
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

source ~/.bashrc
conda activate alphagenome

cd /opt/home/s4076520/alphagenome/scripts/
python finetune_wbins.py
#!/bin/bash
#SBATCH --job-name=rebin_bigwig
#SBATCH --output=/opt/home/s4076520/alphagenome/logs/rebin_%j.txt
#SBATCH --error=/opt/home/s4076520/alphagenome/logs/rebin_%j.err
#SBATCH --partition=SCT
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G

source ~/.bashrc
conda activate alphagenome

python /opt/home/s4076520/alphagenome/scripts/rebin.py
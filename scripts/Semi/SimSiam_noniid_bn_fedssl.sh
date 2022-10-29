#!/bin/bash

#SBATCH --job-name=siam_fedssl_ni_0.1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_new/scripts/slurm/siam_fedssl_noniid_0.1_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="simsiam"
dist="noniid"
iid="False"
norm="bn"
gn="False"
agg="FedSSL"
ema="0.1"
wandb_tag="$exp"_"$agg"_"$dist"_"ema"_"$ema"
ckpt_path=./checkpoints/"$exp"_"$dist"_"$norm"_"$agg".pth.tar
cd /home/kwangyeongill/FedSSL_new/ && python main.py \
                                        --parallel True \
                                        --group_norm $gn \
                                        --exp $exp \
                                        --iid $iid \
                                        --agg $agg \
                                        --wandb_tag $wandb_tag \
                                        --ckpt_path $ckpt_path \
                                        --bn_stat_momentum $ema

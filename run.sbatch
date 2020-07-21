#!/bin/bash

#SBATCH --job-name=morph-age-est-1st-yj450
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yjl450@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load cuda/10.2.89
module load cudnn/10.1v7.6.5.32
module load anaconda3/5.3.1
source deactivate
conda deactivate
conda activate cvtorch
python train.py --data_dir ../ --resume checkpoint/epoch044_0.02343_3.9984.pth --multi_gpu
echo "FINISH"

#!/bin/bash
#SBATCH -J reconstructionlossforCSCMNIST
#SBATCH -p preempt
#SBATCH --gres=gpu:v100:1
#SBATCH --time=30:00:00
#SBATCH -n 24
#SBATCH --mem=20g
#SBATCH --output=class_0.%N.%j.out
#SBATCH --error=class_0.%N.%j.err
#SBATCH --mail-user=thuan.nguyen@tufts.edu
#SBATCH --mail-type=ALL

#[commands_you_would_like_to_exe_on_the_compute_nodes]
module load anaconda
source activate IBM1
python -m sweep_train
source deactivate

#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/svast/u1419542/miniconda3/envs/sentimentClassifierEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate sentimentClassifierEnv

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419542/cs6966/assignment1/models
mkdir -p ${OUT_DIR}

python3 model.py --output_dir ${OUT_DIR}
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=125G
#SBATCH --time=0-03:00
#SBATCH --account=def-seasterb
#SBATCH --mail-user=lemeng.dai@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python
source cgcs/bin/activate
pip install --no-index torch torchvision torchmetrics tensorboard
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun python main.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))

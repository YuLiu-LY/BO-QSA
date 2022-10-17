#!/bin/bash
#SBATCH --job-name=test  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --qos=debug
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

python test_slate.py \
--dataset cars \
--evaluate iou \
--gpus 1 \
--batch_size 256 \
--num_slots 2 \
--log_name embedding_50000_42 \
--test_ckpt_path '/home/liuyu/scratch/SlotAttention/results/cars/embedding_0_seed0/version_0/checkpoints/last.ckpt' \
--seed 42 \
--init_method embedding \
--num_iter 5 \
# --is_logger_enabled \

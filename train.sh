#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --partition=gpu

uv run train_dqn.py
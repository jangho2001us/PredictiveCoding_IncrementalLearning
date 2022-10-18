#!/bin/bash


# pc / pmnist / sgd / mlp /
CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs  50 --seed 0 --eta 0.5
CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 100 --seed 0 --eta 0.5
CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 150 --seed 0 --eta 0.5
CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 200 --seed 0 --eta 0.5

#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs  50 --seed 1 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 100 --seed 1 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 150 --seed 1 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 200 --seed 1 --eta 0.5

#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs  50 --seed 2 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 100 --seed 2 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 150 --seed 2 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 200 --seed 2 --eta 0.5

#CUDA_VISIBLE_DEVICES=0 python run_pc.py --experiment fmnist2 --approach sgd --nepochs  50 --seed 3 --eta 0.5
#CUDA_VISIBLE_DEVICES=0 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 100 --seed 3 --eta 0.5
#CUDA_VISIBLE_DEVICES=0 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 150 --seed 3 --eta 0.5
#CUDA_VISIBLE_DEVICES=0 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 200 --seed 3 --eta 0.5

#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs  50 --seed 4 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 100 --seed 4 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 150 --seed 4 --eta 0.5
#CUDA_VISIBLE_DEVICES=2 python run_pc.py --experiment fmnist2 --approach sgd --nepochs 200 --seed 4 --eta 0.5


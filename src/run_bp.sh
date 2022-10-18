#!/bin/bash

# bp / mnist5 / sgd / mlp
CUDA_VISIBLE_DEVICES=1 python run_bp.py --experiment mnist5 --approach sgd --nepochs 50 --seed 0
CUDA_VISIBLE_DEVICES=1 python run_bp.py --experiment mnist5 --approach sgd --nepochs 50 --seed 1
CUDA_VISIBLE_DEVICES=1 python run_bp.py --experiment mnist5 --approach sgd --nepochs 50 --seed 2
CUDA_VISIBLE_DEVICES=1 python run_bp.py --experiment mnist5 --approach sgd --nepochs 50 --seed 3
CUDA_VISIBLE_DEVICES=1 python run_bp.py --experiment mnist5 --approach sgd --nepochs 50 --seed 4


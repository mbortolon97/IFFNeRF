#!/bin/sh

python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/chair --expname tensorf_chair_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/drums --expname tensorf_drums_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/ficus --expname tensorf_ficus_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/hotdog --expname tensorf_hotdog_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/materials --expname tensorf_materials_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/ship --expname tensorf_ship_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/lego --expname tensorf_lego_VM
python3 train.py --config configs/lego.txt --datadir datasets/nerf_synthetic/mic --expname tensorf_mic_VM
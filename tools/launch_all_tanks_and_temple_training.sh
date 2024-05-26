#!/bin/sh

python3 train.py --config configs/truck.txt --datadir datasets/TanksAndTemple/Barn --expname tensorf_Barn_VMtt
python3 train.py --config configs/truck.txt --datadir datasets/TanksAndTemple/Caterpillar --expname tensorf_Caterpillar_VMtt
python3 train.py --config configs/truck.txt --datadir datasets/TanksAndTemple/Family --expname tensorf_Family_VMtt
python3 train.py --config configs/truck.txt --datadir datasets/TanksAndTemple/Ignatius --expname tensorf_Ignatius_VMtt
python3 train.py --config configs/truck.txt --datadir datasets/TanksAndTemple/Truck --expname tensorf_Truck_VMtt

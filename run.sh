#!bin/bash

data_dir=$1
echo "DATA DIRECTORY PASSED IS $data_dir"
echo "starting data prep"
python3 data_prep.py --data_path=$data_dir
echo "starting feature creation part 1"
python3 feature_creation.py --data_path=$data_dir
echo "starting feature creation part 2"
python3 nm_feature_creation.py --data_path=$data_dir
echo "creating modelling dataset"
python3 feature_join.py --data_path=$data_dir
echo "starting model training"
pip install lightgbm==2.3.1 #sahil's version uses this and nikhil's uses 3.1.1
python3 train.py --data_path=$data_dir
echo "set 1 of models is complete"
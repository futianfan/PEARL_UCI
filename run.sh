#!/bin/bash

### preprocess
python3 data/preprocess_car.py

### rule learning method 
./corels/corels -r 0.0000000015 -c 3 -p 1 data/rule_feature data/rule_label > data/corels_rule_list
python src/rule_learing.py --rule_file ./data/corels_rule_list --test_file ./data/cars/car.data 


## NN-based method
python3 src/train.py





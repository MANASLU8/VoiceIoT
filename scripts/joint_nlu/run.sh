#!/bin/bash

python -m scripts.joint_nlu.normalize
python -m scripts.joint_nlu.split
python -m scripts.joint_nlu.train_and_test
#python -m scripts.joint_nlu.train
#python -m scripts.joint_nlu.test
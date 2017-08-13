#!/bin/bash
now=$(date + '%Y%m%d_%H%M%S')
../caffe/build/tools/caffe train -solver $PROJECTDIR/pretrain/Resnet_pretrain_solver.prototxt -gpu 0 2>&1 | tee logs/log-$now.log

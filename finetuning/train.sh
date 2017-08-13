#!/bin/bash
now=$(date + '%Y%m%d_%H%M%S')
../caffe-augmentation/build/tools/caffe train -weights $PROJECTDIR/pretrain/resnet_pretrain_model.caffemodel -solver $PROJECTDIR/finetuning/Resnet_finetuning_solver.prototxt -gpu 0 2>&1 | tee logs/log-$now.log

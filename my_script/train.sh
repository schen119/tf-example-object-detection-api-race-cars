#!/bin/bash

#object_detection_path=/home/elvis/data/tensorflow/models/research/object_detection
#model_config_path=/home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_bumper.config
#train_saving_dir=/home/elvis/repo/tf_object_detection/data/train


python /home/elvis/data/tensorflow/models/research/object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=/home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_bumper.config \
        --train_dir=/home/elvis/repo/tf_object_detection/data/train


python ${object_detection_path}/train.py \
	--logtostderr \
        --pipeline_config_path=/home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_bumper.config \
        --train_dir=/home/elvis/repo/tf_object_detection/data/train

#python ${object_detection_path}/train.py \ 	
#	--logtostderr \
#	--pipeline_config_path=${model_config_path} \
#	--train_dir=${train_saving_dir}

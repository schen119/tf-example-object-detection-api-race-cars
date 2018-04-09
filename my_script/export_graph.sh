#!bin/bash

python ../export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_bumper.config \
    --trained_checkpoint_prefix /home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/train/model.ckpt-98777 \
    --output_directory /home/elvis/repo/tf_object_detection/models/ssd_mobilenet_v1_coco_2017_11_17/train/bumper_model_graph-98777



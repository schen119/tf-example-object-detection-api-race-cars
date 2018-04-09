#!/bin/bash
python create_pascal_tf_record_onlyXML.py \
        --xml_dir=/home/elvis/repo/labelImg/car/bumper_annotate \
            --output_path=/home/elvis/repo/tf_object_detection/data/xxx.record \
                --label_map_path=/home/elvis/repo/tf_object_detection/data/map.pbtxt

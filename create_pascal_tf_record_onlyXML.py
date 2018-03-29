# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# The script below has been modified to create TFRecord files for
# datasets that use the PASCAL VOC annotation format.
#
# Modifier: Shen-Chi Chen
# This script doesn't have a strict that the annotation file path must be the same as the image file
# The images path are either be extracted from the annotation XML file or assigned by the img_dir
# Accept the 'JPG 'Image format only 
#
# Require input:-- xml_dir: where the annotation XML are
#               -- label_map_path: path where the map.pbtxt is
#               -- img_dir[option]: the img_dir with images. If img_dir isn't given, use the path assigned in xml annotation file  
#
# The original can be found here:
# https://github.com/AndrewCarterUK/tf-example-object-detection-api-race-cars/blob/master/create_pascal_tf_record.py

"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_pascal_tf_record_onlyXML.py \
        --xml_dir=/home/user/dit_with_xml \
        --output_path=/home/user/pascal.record \
        --label_map_path=/home/user/data/map.pbtxt
        --img_dir=[option]path with images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('xml_dir', None, 'Dir with xml annotaion file')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', None,
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_string('img_dir', None, '[option] Dir with images. If no input, use the filed of path in xml annotation file to load image')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       img_dir,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    img_dir: path with image [option], if no input will load the image by the path defined in xml file
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  if img_dir == None:
    image_path = data['path']
  else:
    image_path = os.path.join(img_dir, data['filename'])

  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if not 'object' in data:
      return None

  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  xml_dir = FLAGS.xml_dir

  if not xml_dir:
    logging.error('Must provide a xml annotation directory')
    return

  output_path = FLAGS.output_path

  if not output_path:
    logging.error('Must provide an output path')
    return

  label_map_path = FLAGS.label_map_path

  if not label_map_path:
    logging.error('Must provide a label map path')
    return

  img_dir = FLAGS.img_dir
  if not img_dir:
    logging.warning('No img_dir augment, image will be load from the path assigned in the xml annotation file')

  writer = tf.python_io.TFRecordWriter(output_path)

  label_map_dict = label_map_util.get_label_map_dict(label_map_path)

  logging.info('Reading from data directory.')

  data_dir_xml_query = os.path.join(xml_dir, '*.xml')

  for idx, annotation_path in enumerate(glob.glob(data_dir_xml_query)):
    if idx % 20 == 0:
      logging.info('On annotation file %d (%s)', idx, annotation_path)

    with tf.gfile.GFile(annotation_path, 'r') as fid:
      xml_str = fid.read()

    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, 
                                    label_map_dict,
                                    img_dir,
                                    FLAGS.ignore_difficult_instances)
    if not tf_example == None:
        writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()

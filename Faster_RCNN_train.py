#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : Faster_RCNN_Train.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/19 AM 10:57
# @ Software   : PyCharm
#------------------------------------------------------

import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from Faster_RCNN.faster_rcnn_util import cfgs
from DataProcess.read_coco_pascal_tfrecord import reader_tfrecord
from Faster_RCNN import Faster_RCNN_slim


original_dataset_dir = '/home/alex/Documents/datasets/Pascal_VOC_2012/VOCtrainval/VOCdevkit_test'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecords')
record_file = os.path.join(tfrecord_dir, 'train.tfrecords')

original_pretrain_model_dir = '/home/alex/Documents/pretraing_model/faster_rcnn'
basebone_model_dir = os.path.join(original_pretrain_model_dir, 'resnet_101')
pretrain_model_dir = os.path.join(original_pretrain_model_dir, 'faster_rcnn')

model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'faster_rcnn.pb')
logs_dir = os.path.join(os.getcwd(), 'logs')


tf.app.flags.DEFINE_string('train_dir', record_file, 'Directory to put the training data.')
tf.app.flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
tf.app.flags.DEFINE_string('basebone_model_dir', basebone_model_dir, 'base bone model dir.')
tf.app.flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
tf.app.flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
tf.app.flags.DEFINE_string('model_name', model_name, 'model_name.')
FLAGS = tf.app.flags.FLAGS



if __name__ == "__main__":
    print(FLAGS.pretrain_model_dir)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File DenseNet_Demo.py
# @ Description :
# @ Author alexchung
# @ Time 3/12/2019 PM 16:31

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from Faster_RCNN.Faster_RCNN_Util import cfgs

class FasterRCNN():
    """
    Faster_RCNN
    """
    def __init__(self, base_network_name, is_training):
        self.base_network_name = base_network_name
        self.is_training = is_training
        self.anchor_scale = []
        # self.num_anchor =
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        # self.raw_input_data = tf.compat.v1.placeholder(tf.float32,
        #                                                shape=[None, input_shape[0], input_shape[1], input_shape[2]],
        #                                                name="input_images")
        # # y [None,num_classes]
        # self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        # self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')
        #
        # self.global_step = tf.train.get_or_create_global_step()
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        #
        # # logits
        # self.logits = self.inference(self.raw_input_data, scope='densenet121')
        # # # computer loss value
        # self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # # train operation
        # self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        # self.accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size

    def inference(self, inputs, scope='densenet121'):
        pass

    def faster_rcnn(self):
        pass

    def faster_rcnn_base(self):
        pass
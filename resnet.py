#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : resnet.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/16 AM 09:26
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block


class ResNet():
    def __init__(self, scope_name, weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
                 batch_norm_scale=True):
        self.scope_name = scope_name
        self.weight_decay=weight_decay
        self.batch_norm_decay=batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.fixed_block = 1

    def resnet_arg_scope(self, is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5, batch_norm_scale=True):
        '''
        In Default, do not use BN to train resnet, since batch_size is too small.
        So is_training is False and trainable is False in the batch_norm params.
        '''
        batch_norm_params = {
            'is_training': False, 'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
            'trainable': False,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                trainable=is_training,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc

    def resnet_base(self, inputs, is_training):
        if self.scope_name == 'resnet_v1_50':
            middle_num_units = 6
        elif self.scope_name == 'resnet_v1_101':
            middle_num_units = 23
        else:
            raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....')

        blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                  # use stride 1 for the last conv4 layer.

                  resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
        # when use fpn . stride list is [1, 2, 2]

        with slim.arg_scope(self.resnet_arg_scope(is_training=False)):
            with tf.variable_scope(self.scope_name,'resnet_v1_101'):
                # Do the first few layers manually, because 'SAME' padding can behave inconsistently
                # for images of different sizes: sometimes 0, sometimes 1
                net = resnet_utils.conv2d_same(
                    inputs, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='VALID', scope='pool1')

        # generate freeze flag
        block_freeze = [False] * self.fixed_block + (4-self.fixed_block)*[True]

        with slim.arg_scope(self.resnet_arg_scope(is_training=(is_training and block_freeze[0]))):
            net, _ = resnet_v1.resnet_v1(net,
                                        blocks[0:1],
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=self.scope_name)

        with slim.arg_scope(self.resnet_arg_scope(is_training=(is_training and block_freeze[1]))):
            net, _ = resnet_v1.resnet_v1(net,
                                        blocks[1:2],
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=self.scope_name)
        # add_heatmap(C3, name='Layer/C3')
        # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

        with slim.arg_scope(self.resnet_arg_scope(is_training=(is_training and block_freeze[2]))):
            net, _ = resnet_v1.resnet_v1(net,
                                        blocks[2:3],
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=self.scope_name)
        return net

    def restnet_head(self, inputs, scope_name, is_training):
        block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        with slim.arg_scope(self.resnet_arg_scope(is_training=is_training)):
            net, _ = resnet_v1.resnet_v1(inputs,
                                        block4,
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=scope_name)
            net_flatten = tf.reduce_mean(net, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # global average pooling C5 to obtain fc layers
        return net_flatten






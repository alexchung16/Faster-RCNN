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


tf.app.flags.DEFINE_string('record_file', record_file, 'Directory to put the training data.')
tf.app.flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')

tf.app.flags.DEFINE_string('basebone_model_dir', basebone_model_dir, 'base bone model dir.')
tf.app.flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
tf.app.flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
tf.app.flags.DEFINE_string('model_name', model_name, 'model_name.')
FLAGS = tf.app.flags.FLAGS


def train():
    """
    train progress
    :return:
    """
    faster_rcnn = Faster_RCNN_slim.FasterRCNN(base_network_name='resnet_v1_101', is_training=True)
    #-----------------------------------------------read data----------------------------------------------------------
    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            reader_tfrecord(batch_size=cfgs.BATCH_SIZE,
                            shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                            record_file=FLAGS.record_file,
                            is_training=True)
        gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])


    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d,
                         slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer((cfgs.WEIGHT_DECAY)),
                        biases_regularizer=tf.no_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
        # forward network
        final_bbox, final_scores, final_category, loss_dict = faster_rcnn.faster_rcnn(inputs_batch=img_batch,
                                                                                      gtboxes_batch=gtboxes_and_label)
    #++++++++++++++++++++++++++++++++++++++++++++++++build loss function++++++++++++++++++++++++++++++++++++++++++++++

    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    fastrcnn_cls_loss = loss_dict['fastrcnn_cls_loss']
    fastrcnn_loc_loss = loss_dict['fastrcnn_loc_loss']
    fastrcnn_total_loss = fastrcnn_cls_loss + fastrcnn_loc_loss

    total_loss = rpn_total_loss + fastrcnn_total_loss

    #-----------------------------------------------add summary-------------------------------------------------------
    tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
    tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)

    tf.summary.scalar('LOSS/total_loss', total_loss)

    #-----------------------------------------gegerate optimizer------------------------------------------------------
    global_step = tf.train.get_or_create_global_step()
    # piecewise learning rate
    learning_rate = tf.train.piecewise_constant(global_step,
                                                boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                                values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cfgs.MOMENTUM)
    tf.summary.scalar('learning_rate', learning_rate)

    #-----------------------------------------computer gradient-------------------------------------------------------
    gradients = faster_rcnn.get_gradients(optimizer, total_loss)

    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)

    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients_YJR'):
            gradients = slim.learning.clip_gradient_norms(gradients, cfgs.GRADIENT_CLIPPING_BY_NORM)

    #+++++++++++++++++++++++++++++++++++++++++start train+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=30)

    # support growth train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as sess:
        sess.run(init_op)











if __name__ == "__main__":
    print(FLAGS.pretrain_model_dir)
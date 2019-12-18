#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : losses.py
# @ Description:  
# @ Author     : jemmy li
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/18 PM 15:05
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

def smooth_l1_loss_base(bbox_pred, bbox_target, sigma=1.0):
    """
    sooth loss reference paper(Fast RCNN) formula 3
    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_target: shape is same as bbox_pred
    :param sigma:
    :return:
    """
    sigma_square = sigma ** 2

    box_diff = bbox_pred - bbox_target

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_square)))

    loss_box = tf.pow(box_diff, 2) * (sigma_square / 2.0) * smoothL1_sign  \
               + (abs_box_diff - (0.5 / sigma_square)) * (1.0 - smoothL1_sign)

    return loss_box

def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    """
    rpn bbbox loss reference(Faster RCNN) formula 1
    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    """
    value = smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_mean(value, axis=1)
    rpn_select = tf.where(tf.greater(label, 0)) # remove back_ground

    select_value = tf.gather(value, rpn_select)

    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.cast(tf.equal(label, -1), dtype=tf.float32) # positive is 1.0 other is 0.0
    )

    bbox_loss = tf.reduce_sum(select_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

    return bbox_loss


def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    """
    fast rcnn bbox loss
    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    """
    outside_mask = tf.stop_gradient(tf.cast(tf.greater(label, 0), dtype=tf.float32)) # get positive indices

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = smooth_l1_loss_base(bbox_pred,
                                bbox_targets,
                                sigma=sigma)
    value = tf.reduce_sum(value, axis=2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.cast((tf.shape(bbox_pred)[0]), dtype=tf.float32)
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask,  axis=1) * outside_mask) / normalizer

    return bbox_loss









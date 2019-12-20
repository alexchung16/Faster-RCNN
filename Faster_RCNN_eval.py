#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : Faster_RCNN_eval.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/20 PM 17:31
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import sys
import time
import numpy as np
import cv2 as cv
import tensorflow as tf

from Faster_RCNN.faster_rcnn_util import cfgs
from Faster_RCNN.faster_rcnn_util import draw_box_in_img
from Faster_RCNN.Faster_RCNN_slim import FasterRCNN


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def object_detect(detect_net, image_path):
    """
    detect object
    :param detect_net:
    :param image_path:
    :return:
    """
    pass

def max_length_limitation(length, length_limitation):
    """
    get limitation length
    :param length:
    :param length_limitation:
    :return:
    """
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)

def short_side_resize(img_tensor, target_shortside_len, target_length_limitation=1200):
    '''
    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h,
                                                                  target_length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w,
                                                                   target_length_limitation),
                                             target_shortside_len))
    # expend dimension to 3 for resize
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
    return img_tensor

def image_process(img_batch):
    """
    image_process
    :param image:
    :return:
    """
    img_batch = tf.cast(img_batch, dtype=tf.float32)
    # image resize
    image = short_side_resize(img_tensor=img_batch,
                              target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                              target_length_limitation=cfgs.IMG_MAX_LENGTH)
    # image white
    image = image - tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=tf.float32)

    return image



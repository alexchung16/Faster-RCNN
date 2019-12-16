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

from Faster_RCNN.faster_rcnn_util import cfgs
from Faster_RCNN.resnet import ResNet
from Faster_RCNN.faster_rcnn_util.anchor_utils import make_anchors
from Faster_RCNN.faster_rcnn_util import boxes_utils
from Faster_RCNN.faster_rcnn_util import encode_and_decode


class FasterRCNN():
    """
    Faster_RCNN
    """
    def __init__(self, base_network_name='resnet_v1_101', weight_decay=0.0001, batch_norm_decay=0.997,
                 batch_norm_epsilon=1e-5, batch_norm_scale=True, is_training=False):
        self.base_network_name = base_network_name
        self.weight_decay = weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.is_training = is_training
        self.num_anchors = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.resnet = ResNet(scope_name=self.base_network_name, weight_decay=weight_decay,
                             batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon,
                             batch_norm_scale=batch_norm_scale)

    def inference(self, inputs, scope='densenet121'):
        pass

    def build_base_network(self, input_img_batch):
        if self.base_network_name.startswith('resnet_v1'):
            return self.resnet.resnet_base(input_img_batch, is_training=self.is_training)
        else:
            raise ValueError('Sry, we only support resnet_v1_50 or resnet_v1_101')

    def build_rpn_network(self, inputs_feature):
        """
        build rpn net
        :param inputs_feature:
        :return:
        """
        with tf.variable_scope('build_rpn', regularizer=slim.l2_regularizer(self.weight_decay)):
            rpn_conv_3x3= slim.conv2d(inputs=inputs_feature, num_outputs=512, kernel_size=[3, 3],
                                      weights_initializer=cfgs.INITIALIZER, activation_fn=tf.nn.relu,
                                      trainable=self.is_training, scope='rpn_conv/3x3')

        rpn_cls_score = slim.conv2d(rpn_conv_3x3, self.num_anchors * 2, [1, 1], stride=1, trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER, activation_fn=None, scope='rpn_cls_score')
        rpn_box_pred = slim.conv2d(rpn_conv_3x3, self.num_anchors * 4, [1, 1], stride=1, trainable=self.is_training,
                                   weights_initializer=cfgs.BBOX_INITIALIZER, activation_fn=None, scope='rpn_bbox_pred')
        rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4]) #()
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])  #(background, object)
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')
        return rpn_box_pred, rpn_cls_prob

    def postprocess_rpn_proposals(self, rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
        """
        rpn proposals operation
        :param rpn_bbox_pred: predict bbox
        :param rpn_cls_prob: probability of rpn classification
        :param img_shape: image_shape
        :param anchor: all reference anchor
        :param is_training:
        :return:
        """
        if is_training:
            pre_nms_topN = cfgs.RPN_TOP_K_NMS_TRAIN
            post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
            nms_threshold = cfgs.RPN_NMS_IOU_THRESHOLD
        else:
            pre_nms_topN = cfgs.RPN_TOP_K_NMS_TEST
            post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST
            nms_threshold = cfgs.RPN_NMS_IOU_THRESHOLD

        cls_prob = rpn_cls_prob[:, 1]

        # step 1  decode boxes
        decode_boxes = encode_and_decode.decode_boxes(encoded_boxes=rpn_bbox_pred,
                                                      reference_boxes=anchors,
                                                      scale_factors=cfgs.ANCHOR_SCALE_FACTORS)
        # step 2 clip to image boundaries
        decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes, img_shape=img_shape)
        # step 3 get top N to NMS
        if pre_nms_topN > 0:
            pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='minimum boxes')
            cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
            decode_boxes = tf.gather(params=decode_boxes, indices=top_k_indices)

        # step 4 NMS(Non Max Suppression)
        keep_indices = tf.image.non_max_suppression(boxes=decode_boxes,
                                                    score=cls_prob,
                                                    max_output_size=post_nms_topN,
                                                    iou_threshold=nms_threshold)
        final_boxes = tf.gather(decode_boxes, keep_indices)
        final_probs = tf.gather(cls_prob, keep_indices)
        return final_boxes, final_probs

    def faster_rcnn(self, inputs_batch, gtboxes_batch):
        """
        faster rcnn
        :param input_img_batch:
        :param gtboxes_batch:
        :return:
        """
        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(inputs_batch)
        # step 1 build base network
        feature_cropped = self.build_base_network(inputs_batch)
        # step 2 build rpn
        rpn_box_pred, rpn_cls_prob = self.build_rpn_network(feature_cropped)
        # step 3 make anchor
        feature_height = tf.cast(tf.shape(feature_cropped)[1], dtype=tf.float32)
        feature_width = tf.cast(tf.shape(feature_cropped)[2], dtype=tf.float32)
        # step make anchor
        anchors = make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                               anchor_scales=cfgs.ANCHOR_SCALES,
                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                               feature_height=feature_height,
                               feature_width=feature_width,
                               stride=cfgs.ANCHOR_STRIDE,
                               name='make_anchors_forRPN')

        with tf.variable_scope('postprocess_RPN'):
            rois, rois_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                               rpn_cls_prob=rpn_cls_prob,
                                                               img_shape=img_shape,
                                                               anchors=anchors,
                                                               is_training=self.is_training)


    def faster_rcnn_base(self):
        pass
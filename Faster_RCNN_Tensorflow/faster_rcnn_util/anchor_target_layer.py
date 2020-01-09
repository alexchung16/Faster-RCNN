#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------
# @ File       : anchor_target_layer.py
# @ Description:  
# @ Author     : Ross Girshick and Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/17 PM 14:07
# @ Software   : PyCharm
#---------------------------------------------------

import os
import tensorflow as tf
from Faster_RCNN.faster_rcnn_util import cfgs
import numpy as np
from Faster_RCNN.faster_rcnn_util.cython_utils.cython_bbox import bbox_overlaps
from Faster_RCNN.faster_rcnn_util import encode_and_decode

def anchor_target_layer(gt_boxes, img_shape, all_anchors, is_restrict=False):
    """
    get target anchor the same as Fast/er RCNN
    :param gt_boxes:
    :param img_shape:
    :param all_anchors:
    :param is_restrict:
    :return:
    """
    anchors_num = all_anchors.shape[0]
    img_height, img_width = img_shape[1], img_shape[2]
    gt_boxes = gt_boxes[:, :-1]  # remove class label

    # the number of a small amount boxes allow  to sit over the edge
    allow_border = 0

    # only keep anchors inside the image
    indices_inside = np.where(
        (all_anchors[:, 0] >= -allow_border) &  # left_up_x
        (all_anchors[:, 1] >= -allow_border) &  # left_up_y
        (all_anchors[:, 2] <  img_width + allow_border) &  # right_down_x
        (all_anchors[:, 3] < img_height + allow_border)  # right_down_y
    )[0]

    anchors = all_anchors[indices_inside, :]

    # label: 1 -> positive, 0 -> negative, -1 -> dont care
    labels = np.empty((len(indices_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gtbox
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(indices_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfgs.RPN_IOU_POSITIVE_THRESHOLD] = 1

    if cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0
    # reference paper(Faster RCNN) balance positive and negative ratio

    # num foreground of RPN
    num_fg = int(cfgs.RPN_MINIBATCH_SIZE * cfgs.RPN_POSITIVE_RATE)
    fg_indices = np.where(labels == 1)[0]
    if len(fg_indices) > num_fg:
        disable_indices = np.random.choice(fg_indices, size=(len(fg_indices)-num_fg), replace=False)
        labels[disable_indices] = -1

     # num backgound of RPN
    num_bg = cfgs.RPN_MINIBATCH_SIZE - np.sum(labels==1)
    if is_restrict:
        num_bg = max(num_bg, num_fg * 1.5)

    bg_indices = np.where(labels == 0)[0]
    if len(bg_indices) > num_bg:
        disable_indices = np.random.choice(bg_indices, size=(len(bg_indices) - num_bg), replace=False)
        labels[disable_indices] = -1


    bbox_targets = compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # map up to original set of anchors
    labels = unmap_anchor(labels, anchors_num, indices_inside, fill=-1)
    bbox_targets = unmap_anchor(bbox_targets, anchors_num, indices_inside, fill=0)

    rpn_labels = labels.reshape((-1, 1))

    bbox_targets = bbox_targets.reshape((-1, 4))
    rpn_bbox_targets = bbox_targets

    return rpn_labels, rpn_bbox_targets


def unmap_anchor(data, count, indices, fill=0):
    """
    unmap a set of items data back to the original set of items (of size count)
    :param data:
    :param count:
    :param indices:
    :param fill:
    :return:
    """

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[indices] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[indices, :] = data
    return ret


def compute_targets(ex_rois, gt_rois):
    """
    Compute bound-box regression targets for an image
    :param ex_rois:
    :param gt_rois:
    :return:
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                            reference_boxes=ex_rois,
                                            scale_factors=cfgs.ANCHOR_SCALE_FACTORS)
    return targets



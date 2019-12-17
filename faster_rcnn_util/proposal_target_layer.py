#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : proposal_target_layer.py
# @ Description:  
# @ Author     : Ross Girshick and Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/17 PM 16:48
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np

from Faster_RCNN.faster_rcnn_util import cfgs
from Faster_RCNN.faster_rcnn_util import encode_and_decode
from Faster_RCNN.faster_rcnn_util.cython_utils.cython_bbox import bbox_overlaps

def proposal_target_layer(rpn_roi, gt_boxes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets
    :param rpn_roi: Proposal ROIs (x1, y1, x2, y2) coming from RPN
    :param gt_boxes: gt_boxes (x1, y1, x2, y2, label)
    :return:
    """
    if cfgs.ADD_GTBOXES_TO_TRAIN:
        all_rois = np.vstack(rpn_roi, gt_boxes[:, :-1])
    else:
        all_rois = rpn_roi

    # get rois per image
    rois_per_image = np.inf if cfgs.FAST_RCNN_MINIBATCH_SIZE == -1 else cfgs.FAST_RCNN_MINIBATCH_SIZE

    # number of foreground rois per image
    fg_rois_per_image = np.around(cfgs.FAST_RCNN_POSITIVE_RATE * rois_per_image)

def get_bbox_regression_labels(bbox_target_data, num_classes):
    """

    :param bbox_target_data:
    :param num_classes:
    :return:
    """



def compute_target(ex_rois, gt_rois, labels):
    """
    Compute bounding-box regression targets for an image.
    :param ex_rois:
    :param gt_rois:
    :param lable:
    :return: [label, tx, ty, tw, th]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ROI_SCALE_FACTORS)

    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)




def sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    Generate a random sample of RoIs comprising foreground and background examples.
    :param all_rois: rois shape is [-1, 4]
    :param gt_boxes: gt_boxes shape is [-1, 5]. that is [x1, y1, x2, y2, label]
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_classes: object_classes + 1(background)
    :return:
    """
    # overlaps rois gt_boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :-1], dtype=np.float)
    )
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, -1]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_indices = np.where(max_overlaps >= cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indices = np.where((max_overlaps < cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD) &
                       (max_overlaps >= cfgs.FAST_RCNN_IOU_NEGATIVE_THRESHOLD))[0]

    fg_rois_per_this_image = min(fg_rois_per_image, fg_indices.size)

    # Sample foreground regions without replacement
    if fg_indices.size > 0:
        fg_indices = np.random.choice(fg_indices, size=int(fg_rois_per_this_image), replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_indices.size)
    # Sample background regions without replacement
    if bg_indices.size > 0:
        bg_indices = np.random.choice(bg_indices, size=int(bg_rois_per_this_image), replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_indices, bg_indices)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    # positive -> 1 , negative -> 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]




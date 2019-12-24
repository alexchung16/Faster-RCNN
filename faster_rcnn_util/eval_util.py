#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : eval_util.py
# @ Description:  
# @ Author     : Bharath Hariharan and Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/22 AM 11:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import xml.etree.ElementTree as ET

from Faster_RCNN.faster_rcnn_util import cfgs

NAME_LABEL_MAP = cfgs.PASCAL_NAME_LABEL_MAP


def generate_cls_bbox(all_boxes, img_name_list):
    """

    :param all_boxes: is a list. each item reprensent the detections of a img.
                      the detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
                      Note that: if none detections in this img. that the detetions is : []
    :param img_name_list:
    :return:
    """
    cls_box_dict = {}
    for cls_name, cls_id in NAME_LABEL_MAP.items():
        cls_score_boxes = None
        if cls_id == 0:
            continue
        else:
            for index, img_name in enumerate(img_name_list):
                detection_per_img = all_boxes[index]
                detection_cls_per_img = detection_per_img[detection_per_img[:, 0].astype(np.int32) == cls_id]
                if detection_cls_per_img.shape[0] == 0: # this cls has none detections in this image
                    continue
                else:
                    img_index = np.array([[index] * detection_cls_per_img.shape[0]]).transpose()
                    cls_box_score_per_img = np.hstack((img_index, detection_cls_per_img[:, 1:]))
                if cls_score_boxes is None:
                    cls_score_boxes = cls_box_score_per_img
                else:
                    cls_score_boxes = np.vstack((cls_score_boxes, cls_box_score_per_img))
        cls_box_dict[cls_name] = cls_score_boxes

    return cls_box_dict

def parse_rec(filename):
  """
  Parse a PASCAL VOC xml file
  :param filename:
  :return:
  """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects

def voc_ap(rec, prec):
    """
    ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    :param rec:
    :param prec:
    :return:
    """
    pass

def voc_eval(cls_box_dict, annotation_path, img_name_list, cls_name, ovthresh=0.5, use_diff=False):
    """
    calculate  precision, recall and mAP per class
    :param detpath:
    :param annopath:
    :param test_imgid_list:
    :param cls_name:
    :param ovthresh:
    :param use_diff:
    :return:
    """


def execute_eval(img_name_list, cls_box_dict, annotation_path):
    """
    execute evaluation
    :param img_name_list:
    :param cls_box_dict:
    :param annotation_path:
    :return:
    """


def voc_evaluate_detection(all_boxes, annotation_path, img_name_list):
    """

    :param all_boxes: is a list. each item represent the detections of a img.
                      The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
                      Note that: if none detections in this img. that the detetions is : []
    :param annotation_path:
    :param image_list:
    :return:
    """
    img_id_list = [img_name.split('.')[0] for img_name in img_name_list]

    cls_box_dict = generate_cls_bbox(all_boxes=all_boxes, img_name_list=img_id_list)

    execute_eval(img_name_list=img_id_list,
                 cls_box_dict=cls_box_dict,
                 annotation_path=annotation_path)



if __name__ == "__main__":
    pass

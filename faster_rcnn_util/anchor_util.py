#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File faster_anchor_util.py
# @ Description : faster_rcnn config
# @ Author alexchung
# @ Time 16/12/2019 PM 15:03

import tensorflow as tf

def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, feature_height, feature_width,
                stride, name='make_anchors'):
    """

    :param base_anchor_size:
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :param name:
    :return:
    """
    with tf.variable_scope(name, default_name='make_anchor'):
        # [x_center, y_center, w, h]
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)

        # (-1, len(anchor_scales)*len(anchor_ratios)), (-1, len(anchor_scales)*len(anchor_ratios))
        w_scale, h_scale = enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)
        x_centers = tf.range(feature_width, dtype=tf.float32) * stride
        y_centers = tf.range(feature_height, dtype=tf.float32) * stride

        # broadcast center coordinate
        # (len(feature_height), len(feature_width))
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        # (len(feature_height))*(len(feature_width), len(anchor_scales)*len(anchor_ratios))
        w_scale, x_centers = tf.meshgrid(w_scale, x_centers)
        h_scale, y_centers = tf.meshgrid(h_scale, y_centers)

        # (len(feature_height))*(len(feature_width), len(anchor_scales)*len(anchor_ratios), 2)
        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([w_scale, h_scale], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])

        anchors = tf.concat(values=[anchor_centers - 0.5*box_sizes, anchor_centers + 0.5*box_sizes], axis=1)

        return anchors


def enum_scales(base_anchor, anchor_scales):
    """
    enumerate scale size
    :param base_anchor:
    :param anchor_scales:
    :return: anchor_scale->shape=((anchor_scales), len(base_anchor))
    """
    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))
    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    """
    ratio = h_scale / w_scale
    :param anchors:
    :param anchor_ratio:
    :return:
    """
    w_scale = anchors[:, 2]  # for base anchor: w == h
    h_scale = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    w_scale = tf.reshape(w_scale / sqrt_ratios[:, tf.newaxis], shape=(-1, 1))
    h_scale = tf.reshape(h_scale * sqrt_ratios[:, tf.newaxis], shape=(-1, 1))
    return w_scale, h_scale


if __name__ == "__main__":
    base_anchor_size = 256
    anchor_scales = [0.5, 1.0, 2.0]
    anchor_ratios = [0.5, 1.0, 2.0]
    feature_height = 280
    feature_width = 300
    stride = [16]
    base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)

    anchors = make_anchors(base_anchor_size=base_anchor_size, anchor_scales=anchor_scales, anchor_ratios=anchor_ratios,
                           feature_height=feature_height, feature_width=feature_width, stride=stride)
    anchor_scales = enum_scales(base_anchor, anchor_scales)

    w_scale, h_scale = enum_ratios(anchor_scales, anchor_ratios)

    with tf.Session() as sess:
        print(sess.run(anchor_scales))
        print(sess.run([w_scale, h_scale]))
        print(sess.run(anchors))




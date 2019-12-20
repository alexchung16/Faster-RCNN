#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File DenseNet_Demo.py
# @ Description :
# @ Author alexchung
# @ Time 3/12/2019 PM 16:31

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from Faster_RCNN.faster_rcnn_util import cfgs
from Faster_RCNN.resnet import ResNet
from Faster_RCNN.faster_rcnn_util.anchor_utils import make_anchors
from Faster_RCNN.faster_rcnn_util import boxes_utils
from Faster_RCNN.faster_rcnn_util import encode_and_decode
from Faster_RCNN.faster_rcnn_util import show_box_in_tensor
from Faster_RCNN.faster_rcnn_util.anchor_target_without_boxweight import anchor_target_layer
from Faster_RCNN.faster_rcnn_util.proposal_target_layer import proposal_target_layer
from Faster_RCNN.faster_rcnn_util import losses


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

        # x [img_height, img_height, img_width]
        self.raw_input_data = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 3], name="input_images")
        # y [None, upper_left_x, upper_left_y, down_right_x, down_right_y]
        self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name="gtbox_label")
        # # is_training flag
        # self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

    def build_base_network(self, input_img_batch):
        if self.base_network_name.startswith('resnet_v1'):
            return self.resnet.resnet_base(input_img_batch, is_training=self.is_training)
        else:
            raise ValueError('Sry, we only support resnet_50 or resnet_101')

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
        # (img_height * img_width * mum_anchor, 4)
        # 4 => (t_center_x, t_center_y, t_width, t_height)
        rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
        # (img_height*img_width*mum_anchor, 2)
        # 2 => (background_prob, object_prob)
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

        return rpn_box_pred, rpn_cls_score

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

        cls_prob = rpn_cls_prob[:, 1] #(, 2) =>（negtive, postive）

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

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, img_shape):
        '''
        generate target box and label
        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            # remove background(index_num=0) just generate object boxes and label
            for i in range(1, cfgs.CLASS_NUM + 1):
                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping'):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            # Stops gradient computation
            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(image=feature_maps,
                                                            boxes=normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ], dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )
            # (cfgs.ROI_SIZE, cfgs.ROI_SIZE) =>  cfgs.FAST_RCNN_MINIBATCH_SIZE x 14 x 14 x 1024
            rois_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)
            # cfgs.FAST_RCNN_MINIBATCH_SIZE x 7 x 7 x 1024
            return rois_features

    def build_fastrcnn(self, feature_crop, rois, img_shape):
        """
        build fastrcnn
        !batch_size =  cfgs.FAST_RCNN_MINIBATCH_SIZE!
        :param feature_ro_crop: feature map
        :param rois:
        :param img_shape:
        :return:
        """
        with tf.variable_scope('Fast-RCNN'):
            # step 5 ROI Pooling
            with tf.variable_scope('roi_pooling'):
                pooled_feature = self.roi_pooling(feature_maps=feature_crop, rois=rois, img_shape=img_shape)
            # step 6 Inference rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):

                # cfgs.FAST_RCNN_MINIBATCH_SIZE x 2048
                fc_flatten = self.resnet.restnet_head(inputs=pooled_feature,
                                                      is_training=self.is_training,
                                                      scope_name=self.base_network_name)
            else:
                raise NotImplementedError('only support resnet_50 and resnet_101')

            # cls and reg in Fast-RCNN
            with slim.arg_scope([slim.fully_connected], weight_regularizer=slim.l2_regularizer(self.weight_decay)):
                # cfgs.FAST_RCNN_MINIBATCH_SIZE x cfgs.CLASS_NUM + 1
                cls_score = slim.fully_connected(fc_flatten,
                                                 num_outputs=cfgs.CLASS_NUM + 1,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                 activation_fn=None,
                                                 trainable=self.is_training,
                                                 scope='cls_fc')
                # cfgs.FAST_RCNN_MINIBATCH_SIZE x ((cfgs.CLASS_NUM + 1) * 4)
                bbox_pred = slim.fully_connected(fc_flatten,
                                                 num_outputs=(cfgs.CLASS_NUM + 1) * 4,
                                                 weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                                       mode='FAN_AVG',
                                                                                                       uniform=True),
                                                 activation_fn=None,
                                                 trainable=self.is_training,
                                                 scope='reg_fc')

                cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM+1])
                bbox_pred = tf.reshape(bbox_pred, [-1, 4*(cfgs.CLASS_NUM+1)])

                return bbox_pred, cls_score

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels, bbox_pred, bbox_targets,
                   cls_score, labels):
        """
        loss function
        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        :return:
        """
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                # get bbox losses(localization loss)
                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                            bbox_targets=rpn_bbox_targets,
                                                            labels=rpn_labels,
                                                            sigma=cfgs.RPN_SIGMA)
                # select foreground and background
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), shape=[-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), shape=[-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), shape=[-1])

                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(features=rpn_cls_score,
                                                                                              labels=rpn_labels))
                #------------------------------ RPN classification and localization loss-------------------------------
                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)
                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score,
                        labels=labels))  # because already sample before

                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                               labels=labels,
                                                               bbox_targets=bbox_targets,
                                                               bbox_pred=bbox_pred,
                                                               num_ohem_samples=256,
                                                               num_classes=cfgs.CLASS_NUM + 1)

                # ----------------------- Faster RCNN classification and localization loss------------------------------
                cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT

            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss': cls_loss,
                'fastrcnn_loc_loss': bbox_loss
            }
        return loss_dict

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
        rpn_box_pred, rpn_cls_score = self.build_rpn_network(feature_cropped)
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')
        # step 3 make anchor
        feature_height = tf.cast(tf.shape(feature_cropped)[1], dtype=tf.float32)
        feature_width = tf.cast(tf.shape(feature_cropped)[2], dtype=tf.float32)
        # step make anchor
        # reference anchor coordinate
        # (img_height*img_width*mum_anchor, 4)
        #++++++++++++++++++++++++++++++++++++generate anchors+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        anchors = make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                               anchor_scales=cfgs.ANCHOR_SCALES,
                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                               feature_height=feature_height,
                               feature_width=feature_width,
                               stride=cfgs.ANCHOR_STRIDE,
                               name='make_anchors_forRPN')
        # step 4 postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_RPN'):
            rois, roi_scores = self.postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                               rpn_cls_prob=rpn_cls_prob,
                                                               img_shape=img_shape,
                                                               anchors=anchors,
                                                               is_training=self.is_training)
            # +++++++++++++++++++++++++++++++++++++add img summary++++++++++++++++++++++++++++++++++++++++++++++++++++
            # if self.is_training:
            #     rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=inputs_batch,
            #                                                             boxes=rois,
            #                                                             scores=roi_scores)
            #     tf.summary.image('all_rpn_rois', rois_in_img)
            #
            #     score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
            #     score_gre_05_rois = tf.gather(rois, score_gre_05)
            #     score_gre_05_score = tf.gather(roi_scores, score_gre_05)
            #     score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=inputs_batch,
            #                                                                     boxes=score_gre_05_rois,
            #                                                                     scores=score_gre_05_score)
            #     tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
        #++++++++++++++++++++++++++++++++++++++++get rpn_lablel and rpn_bbox_target++++++++++++++++++++++++++++++++++++
        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                rpn_labels, rpn_box_targets = tf.py_func(func=anchor_target_layer,
                                                        inp=[gtboxes_batch, img_shape, anchors],
                                                        Tout=[tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_box_targets, shape=(-1, 4))

                rpn_labels = tf.cast(rpn_labels, dtype=tf.int32, name='to_int32')
                rpn_labels = tf.reshape(rpn_labels, shape=[-1])

            #+++++++++++++++++++++++++++++++++++generate target boxes and labels++++++++++++++++++++++++++++++++++++++++
            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            # get positive and negative indices and ignore others where rpn label value equal to -1
            kept_rpn_indices = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), shape=[-1])
            rpn_cls_category = tf.gather(rpn_cls_category, indices=kept_rpn_indices)
            rpn_cls_labels = tf.cast(tf.gather(rpn_labels, indices=kept_rpn_indices), dtype=tf.int64)
            # evaluate function
            acc = tf.cast(tf.reduce_mean(tf.equal(rpn_cls_category, rpn_cls_labels)), dtype=tf.float32)
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, bbox_targets = tf.py_func(proposal_target_layer,
                                                            [rois, gtboxes_batch],
                                                            [tf.float32, tf.float32, tf.float32])
                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.cast(labels, dtype=tf.int32)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 4*(cfgs.CLASS_NUM + 1)])

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # step 5 build fast-RCNN
        bbox_pred, cls_score = self.build_fastrcnn(feature_crop=feature_cropped, rois=rois, img_shape=img_shape)

        cls_prob = slim.softmax(cls_score, 'cls_prob')

        #----------------------------------------------add smry---------------------------------------------------------
        if self.is_training:
            cls_category = tf.argmax(cls_prob)
            cls_labels = tf.cast(labels, dtype=tf.int64)
            fast_acc = tf.reduce_mean(tf.cast(tf.equal(cls_category, cls_labels), dtype=tf.float32))
            tf.summary.scalar('ACC/fast_acc', fast_acc)

        # step 6 postprocess fastrcnn
        if not self.is_training:
            return self.postprocess_fastrcnn(rois, bbox_ppred=bbox_pred, scores=cls_prob, img_shape=img_shape)
        else:
            '''
            when train, we need to build loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred=bbox_pred,
                                        bbox_targets=bbox_targets,
                                        cls_score=cls_score,
                                        labels=labels)

            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=rois,
                                                                                 bbox_ppred=bbox_pred,
                                                                                 scores=cls_prob,
                                                                                 img_shape=img_shape)
            return final_bbox, final_scores, final_category, loss_dict

    def get_gradients(self, optimizer, loss):
        """
        compute gradient
        :param optimizer:
        :param total_loss:
        :return:
        """
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):
        """

        :param gradients:
        :return:
        """
        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients

    def get_restore(self, pretrain_model_dir, restore_from_rpn=True, is_pretrain=False):
        """
        restore pretrain weight
        :param pretrain_model_dir:
        :param is_pretrain:
        :return:
        """
        faster_rcnn_dir = os.path.join(pretrain_model_dir, 'faster_rcnn')

        base_net_dir = os.path.join(pretrain_model_dir, self.base_network_name)

        model_variables = slim.get_model_variables()
        # restore weight of base net(resnet_50, resnet_v1_101) and rpn_net
        if is_pretrain:
            if restore_from_rpn:
                restore_variables= [var for var in model_variables if not var.name.startwith('Fast-RCNN')]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            checkpoint_path = tf.train.latest_checkpoint(faster_rcnn_dir)

        # restore variable weight only from base_net(resnet_v1_50, resnet_v1_101)
        else:
            ckpt_var_dict = {}
            for var in model_variables:
                if var.name.startwith(self.base_network_name):
                    var_name_ckpt = var.name
                    ckpt_var_dict[var_name_ckpt] = var
            restore_variables = ckpt_var_dict
            for key, item in restore_variables:
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)

            restorer = tf.train.Saver(restore_variables)
            checkpoint_path = os.path.join(base_net_dir, self.base_network_name + '.ckpt')
            print("restore from pretrained_weighs in IMAGE_NET")

        return restorer, checkpoint_path















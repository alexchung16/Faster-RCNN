#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : Faster_RCNN_inference.py
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


class ObjectInference():
    def __init__(self, base_network_name, pretrain_model_dir):
        self.base_network_name = base_network_name
        self.pretrain_model_dir = pretrain_model_dir
        self.detect_net = FasterRCNN(base_network_name=base_network_name, is_training=False)
        # self._R_MEAN = 123.68
        # self._G_MEAN = 116.779
        # self._B_MEAN = 103.939

    def exucute_detect(self, image_path):
        """
        execute object detect
        :param detect_net:
        :param image_path:
        :return:
        """
        inputs_img = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3), name='image_inputs')

        # img_shape = tf.shape(inputs_img)
        # image resize and white process
        image = self.image_process(inputs_img)
        # expend dimension
        img_batch = tf.expand_dims(input=image, axis=0) # (1, None, None, 3)

        # load detect network
        detection_boxes, detection_scores, detection_category = self.detect_net.faster_rcnn(
            inputs_batch=img_batch,
            gtboxes_batch=None)

        # config gpu to growth train
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session() as sess:
            sess.run(init_op)

            # restore pretrain weight
            restorer, restore_ckpt = self.detect_net.get_restore(pretrain_model_dir=self.pretrain_model_dir,
                                                                 restore_from_rpn=False,
                                                                 is_pretrain=True)
            if not restorer is None:
                restorer.restore(sess, save_path=restore_ckpt)
                print('Successful restore model from {0}'.format(restore_ckpt))

            # construct image path list
            format_list = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
            if os.path.isfile(image_path):
                image_list = [image_path]
            else:
                image_list = [os.path.join(image_path, img_name) for img_name in os.listdir(image_path)
                              if img_name.endswith(format_list) and os.path.isfile(os.path.join(image_path, img_name))]

            assert len(image_list) != 0
            print("test_dir has no imgs there. Note that, we only support img format of {0}".format(format_list))
            #+++++++++++++++++++++++++++++++++++++start detect+++++++++++++++++++++++++++++++++++++++++++++++++++++=++
            for index, img_name in enumerate(image_list):
                raw_img = cv.imread(img_name)
                start_time = time.time()
                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run(
                        [img_batch, detection_boxes, detection_scores, detection_category],
                        feed_dict={inputs_img: raw_img[:, :, ::-1]}  # convert channel from BGR to RGB (cv is BGR)
                    )
                end_time = time.time()
                print("{} cost time : {} ".format(img_name, (end_time - start_time)))

                show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores)
                return final_detections


    def image_process(self, img):
        """
        image_process
        :param image:
        :return:
        """
        image = tf.cast(img, dtype=tf.float32)
        # image resize
        image = self.short_side_resize(img_tensor=image,
                                       target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                       target_length_limitation=cfgs.IMG_MAX_LENGTH)
        # image white
        image = image - tf.constant(cfgs.PIXEL_MEAN)

        return image

    def short_side_resize(self, img_tensor, target_shortside_len, target_length_limitation=1200):
        '''
        :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
        :param target_shortside_len:
        :param length_limitation: set max length to avoid OUT OF MEMORY
        :return:
        '''
        img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
        new_h, new_w = tf.cond(tf.less(img_h, img_w),
                               true_fn=lambda: (target_shortside_len,
                                                self.max_length_limitation(target_shortside_len * img_w // img_h,
                                                                      target_length_limitation)),
                               false_fn=lambda: (self.max_length_limitation(target_shortside_len * img_h // img_w,
                                                                       target_length_limitation),
                                                 target_shortside_len))
        # expend dimension to 3 for resize
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
        # recover dimension
        img_tensor = tf.squeeze(img_tensor, axis=0)

        return img_tensor

    def max_length_limitation(self, length, length_limitation):
        """
        get limitation length
        :param length:
        :param length_limitation:
        :return:
        """
        return tf.cond(tf.less(length, length_limitation),
                       true_fn=lambda: length,
                       false_fn=lambda: length_limitation)


if __name__ == "__main__":
    base_network_name = 'resnet_v1_101'
    image_dir = '/home/alex/python_code/eval_test/demo_image'
    image_path = os.path.join(image_dir,'004640.jpg')
    pretrain_model_dir = '/home/alex/Documents/pretraing_model/faster_rcnn'
    inference = ObjectInference(base_network_name=base_network_name, pretrain_model_dir=pretrain_model_dir)

    object_img = inference.exucute_detect(image_path)
    print(object_img)
    cv.imshow('object_name', object_img[:, :, ::-1])
    cv.waitKey()










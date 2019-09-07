# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Vscode
#   File name   : dataset.py
#   Author      : 
#   Created date: 
#   Description :
#
#================================================================

import cv2
import numpy as np
from core import utils
import tensorflow as tf
from core import Config
class Parser(object):
    def __init__(self,tag=1, debug=False):

        self.anchors     = Config.ANCHORS
        self.num_classes = Config.NUM_CLASSES
        self.image_h     = Config.IMAGE_H 
        self.image_w     = Config.IMAGE_W 
        self.tag         = tag
        self.debug       = debug

    def flip_left_right(self, image, gt_boxes):

        w = tf.cast(tf.shape(image)[1], tf.float32)
        image = tf.image.flip_left_right(image)

        xmin, ymin, xmax, ymax, label = tf.unstack(gt_boxes, axis=1)
        xmin, ymin, xmax, ymax = w-xmax, ymin, w-xmin, ymax
        gt_boxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        return image, gt_boxes

    def random_distort_color(self, image, gt_boxes):

        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        return image, gt_boxes

    def random_blur(self, image, gt_boxes):

        gaussian_blur = lambda image: cv2.GaussianBlur(image, (5, 5), 0)
        h, w = image.shape.as_list()[:2]
        image = tf.py_func(gaussian_blur, [image], tf.uint8)
        image.set_shape([h, w, 3])

        return image, gt_boxes

    def preprocess(self, image, gt_boxes):

        ################################# data augmentation ##################################
        is_flip    = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        is_distort = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        is_blur    = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        image, gt_boxes = tf.cond(tf.equal(is_flip, 1),lambda:self.flip_left_right(image, gt_boxes),lambda:(image, gt_boxes))
        image, gt_boxes = tf.cond(tf.equal(is_distort, 1),lambda:self.random_distort_color(image, gt_boxes),lambda:(image, gt_boxes))
        image, gt_boxes = tf.cond(tf.equal(is_blur, 1),lambda:self.random_blur(image, gt_boxes),lambda:(image, gt_boxes))

        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.image_h, self.image_w)

        if self.debug: return image, gt_boxes

        y_true = tf.py_func(self.preprocess_true_boxes, inp=[gt_boxes],
                            Tout = tf.float32)
        image = image / 255.

        return image, y_true

    def preprocess_true_boxes(self, gt_boxes):
        
        if self.tag == 1:
            grid_stride = 32
        else:
            grid_stride = 8
        grid_sizes  = [self.image_h // grid_stride, self.image_w // grid_stride]
        
        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
        box_sizes   = gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        y_true = np.zeros(shape=[grid_sizes[0], grid_sizes[1], 3, 5+self.num_classes], dtype=np.float32)

        wh = box_sizes
        wh = np.expand_dims(wh,-2)

        intersect = np.minimum(wh,self.anchors)
        box_area = wh[..., 0] * wh[..., 1]
        intersect_area = intersect[..., 0] * intersect[..., 1]
        anchor_area = self.anchors[..., 0] * self.anchors[..., 1]
        iou =intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            i = np.floor(gt_boxes[t,0]/self.image_w*grid_sizes[1]).astype('int32')
            j = np.floor(gt_boxes[t,1]/self.image_h*grid_sizes[0]).astype('int32')
            c = gt_boxes[t, 4].astype('int32')

            y_true[j, i, n, 0:4] = gt_boxes[t, 0:4]
            y_true[j, i, n,   4] = 1.
            y_true[j, i, n, 5+c] = 1.

        return y_true

    def parser_example(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image' : tf.FixedLenFeature([], dtype = tf.string),
                'boxes' : tf.FixedLenFeature([], dtype = tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1,5])

        return self.preprocess(image, gt_boxes)

class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None, repeat=True):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat  = repeat
        self._buildup()

    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(map_func = self.parser.parser_example, num_parallel_calls = 10)
        self._TFRecordDataset = self._TFRecordDataset.repeat() if self.repeat else self._TFRecordDataset

        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)

        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()

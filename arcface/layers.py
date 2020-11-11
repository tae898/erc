"""MIT License

Copyright (c) 2019 Kuan-Yu Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

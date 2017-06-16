#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import os
import multiprocessing

import cv2
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import *
import tensorpack.utils.logger as logger
from data_provider import get_dump_data
import video
import p2dist.p2dist as p2dist

TOTAL_BATCH_SIZE = 32
INPUT_SHAPE = 224
DEPTH = None

"""
As our main focus is not model selection, we use the common ResNet-50 architecture for our task.
"""


class OnlineCorrelationMatrix(Callback):
    """From our paper, we know that this specific video pair should have a matching tour.
    So we dump the correlation matrix to tensorboard.
    """
    def __init__(self):
        self.left = "/graphics/scratch/wieschol/sync-data/2012-06-22/3D_L0004_noaudio.low.mp4"
        self.right = "/graphics/scratch/wieschol/sync-data/2012-07-16/3D_L0004_noaudio.low.mp4"
        self.skip = 10
        self.batch = 32
        self.cc = 0

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(['input_frames'], ['encoding'])

        # compute correlation without SVD-decorrelation
        self.plhdr_a = tf.placeholder(dtype=tf.float32)
        self.plhdr_b = tf.placeholder(dtype=tf.float32)

        correlation = p2dist.p2dist(self.plhdr_a, tf.transpose(self.plhdr_b, [0, 2, 1]))
        correlation = tf.log(2 * tf.pow(correlation, 8) + 1)

        correlation = correlation - tf.reduce_min(correlation)
        correlation = correlation / tf.reduce_max(correlation)
        self.correlation = correlation * 255.

    def _embed_video(self, fn):
        vid = video.Reader(fn)
        embeddings = np.zeros((vid.frames // self.skip, 1000))
        offset = 0
        for frames, num in vid.batch_reader(self.batch, self.skip, resize=(224, 224)):
            batch_embedding = self.pred([frames])[0]
            print batch_embedding.sum()
            embeddings[offset:offset + num, ...] = batch_embedding
            offset += num

        return np.expand_dims(embeddings, axis=0)

    def _trigger_epoch(self):

        emb_left = self._embed_video(self.left)
        emb_right = self._embed_video(self.right)

        cor = self.pred.sess.run(self.correlation, {self.plhdr_a: emb_left, self.plhdr_b: emb_right})

        print "cor.sum()", cor.sum()

        self.trainer.monitors.put_image('correlation', cor[0])
        p = os.path.join(logger.LOG_DIR, 'correlation%i.jpg' % self.cc)
        cv2.imwrite(p, cor[0])
        self.cc += 1


def normalize(x, eps=1e-12):
    def l2_norm(t, eps=1e-12):
        return tf.sqrt(tf.reduce_sum(tf.square(t), 1, True) + eps)
    return x / l2_norm(x, eps=eps)


class Model(ModelDesc):
    def __init__(self, data_format='NCHW', depth=18):
        if data_format == 'NCHW':
            assert tf.test.is_gpu_available()
        self.data_format = data_format
        self.depth = depth

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'anchor'),
                InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'positive'),
                InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'negative')]

    def _build_graph(self, inputs):
        anc, pos, neg = inputs

        inputs = tf.concat([anc, pos, neg], axis=0)
        inputs = tf.cast(inputs, tf.float32) * (1.0 / 255)
        inputs = tf.placeholder_with_default(inputs, shape=[None, INPUT_SHAPE, INPUT_SHAPE, 3], name='input_frames')

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        inputs = (inputs - image_mean) / image_std

        if self.data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[self.depth]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope(BatchNorm, use_local_stat=True):
            logits = (LinearWrap(inputs)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .BNReLU('bnlast')
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000, nl=tf.identity)())

        tf.identity(logits, name='encoding')
        encodings = tf.identity(normalize(logits), name='normalized_encoding')
        anc_enc, pos_enc, neg_enc = tf.split(encodings, 3, axis=0)

        wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(wd_cost)

        loss, pos_dist, neg_dist = symbf.triplet_loss(anc_enc, pos_enc, neg_enc, 0.5, extra=True, scope="loss")
        self.cost = tf.add_n([loss, wd_cost], name='cost')
        add_moving_summary(pos_dist, neg_dist, self.cost, loss)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    df = get_dump_data()
    df = PrefetchDataZMQ(df, min(20, multiprocessing.cpu_count()))
    df = BatchData(df, BATCH_SIZE)
    return df


def get_config(data_format='NCHW', depth=18):
    dataset_train = get_data('train')

    return TrainConfig(
        model=Model(data_format=data_format, depth=depth),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
            OnlineCorrelationMatrix()
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(["tower0/loss/pos-dist", "tower0/loss/neg-dist", "tower0/cost"]),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=5000,
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[18, 34, 50, 101])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU // 3  # we use (anchor, pos, neg)

    logger.auto_set_dir()
    config = get_config(data_format=args.data_format, depth=args.depth)
    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()

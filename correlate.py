import h5py
import tensorflow as tf
from user_ops import p2dist
import numpy as np
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', help='path to h5 embeddings', type=str, required=True)
    parser.add_argument('--right', help='path to h5 embeddings', type=str, required=True)
    args = parser.parse_args()

    assert args.left.endswith('.h5')
    assert args.right.endswith('.h5')

    left = np.expand_dims(h5py.File(args.left, 'r')['group1']['dataset1'], axis=0)
    right = np.expand_dims(h5py.File(args.right, 'r')['group1']['dataset1'], axis=0)

    plhdr_a = tf.placeholder(dtype=tf.float32)
    plhdr_b = tf.placeholder(dtype=tf.float32)

    correlation = p2dist.p2dist(plhdr_a, tf.transpose(plhdr_b, [0, 2, 1]))
    correlation = tf.log(2 * tf.pow(correlation, 8) + 1)

    correlation = correlation - tf.reduce_min(correlation)
    correlation = correlation / tf.reduce_max(correlation)
    correlation = correlation * 255.

    with tf.Session() as sess:
        cor = sess.run(correlation, {plhdr_a: left, plhdr_b: right})

    print cor[0].shape
    cv2.imwrite('cor.jpg', cor[0])

    print cor.shape

# manually generated file
import tensorflow as tf
import os

__all__ = ['p2dist']

path = os.path.join(os.path.dirname(__file__), 'p2dist_op.so')
_p2dist_module = tf.load_op_library(path)
p2dist = _p2dist_module.p2dist

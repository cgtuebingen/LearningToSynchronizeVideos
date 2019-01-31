# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops  # noqa


__all__ = []


def loadOps(name, grad=False):
    path = os.path.join(os.path.dirname(__file__), '%s_op.so' % name)
    loaded_modul = tf.load_op_library(path)

    if grad:
        # forward, backward
        return getattr(loaded_modul, name), getattr(loaded_modul, name + '_grad')
    else:
        # forward
        return getattr(loaded_modul, name)


__all__ += ['p2dist']
p2dist = loadOps('p2dist')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

"""
Just a wrapper to ease the process of handling videos in OpenCV.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


class Reader(object):
    """Read Videos"""
    def __init__(self, video_path):
        super(Reader, self).__init__()
        assert os.path.isfile(video_path), "The video %s does not exists!" % video_path

        self.video_path = video_path
        self.vid = cv2.VideoCapture(video_path)
        # print("read video {}\n  fps: {}\n  frames: {}\n  shape: {}".format(video_path, self.fps,
        #                                                                    self.frames, self.shape))

    @property
    def fps(self):
        return self.vid.get(cv2.CAP_PROP_FPS)

    @property
    def frames(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def pos(self):
        return int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def height(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def total_seconds(self):
        return self.frames / self.fps

    @property
    def shape(self):
        w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (h, w)

    def read(self, scale=None):
        _, frame = self.vid.read()
        if scale:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return frame

    def jump(self, frame=None, ms=None):
        assert frame is not ms, "Use either frame or ms, not both!"
        if frame:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if ms:
            self.vid.set(cv2.CAP_PROP_POS_MSEC, ms)
        # print("jumped to frame %i" % self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    def batch_reader(self, batch=32, skip=1, resize=None):
        """Read from current position batches
        """

        pos = self.pos
        num_frames = self.frames

        length = (num_frames - pos) // skip
        num_batches = length // batch

        if resize:
            resize_shape = list(resize)
        else:
            resize_shape = list(self.shape)

        first_batch = 0 if num_batches * batch == length else length - num_batches * batch
        if first_batch > 0:
            batched_frames = np.zeros([first_batch] + resize_shape + [3])
            for i in xrange(first_batch):
                frm = self.read()
                if resize:
                    frm = cv2.resize(frm, resize)
                batched_frames[i, ...] = frm
                for _ in range(skip - 1):
                    self.read()
            yield [batched_frames, first_batch]

        for _ in tqdm(xrange(num_batches)):
            batched_frames = np.zeros([batch] + resize_shape + [3])
            for i in xrange(batch):
                frm = self.read()
                if resize:
                    frm = cv2.resize(frm, resize)
                batched_frames[i, ...] = frm
                for _ in range(skip - 1):
                    self.read()
            yield [batched_frames, batch]


class Writer(object):
    """Write Videos"""
    def __init__(self, fn, height, width, channels=3, fps=30):
        super(Writer, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.fn = fn
        print("writing to %s with shape (%i, %i, %i)" % (fn, height, width, channels))

        cc4 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.out = cv2.VideoWriter(fn, cc4, fps, (width, height), isColor=(channels == 3))

    def write(self, frame):
        if len(frame.shape) == 2:
            frame = frame[:, :, None]

        assert frame.shape == (self.height, self.width, self.channels), "{} vs {}".format(frame.shape,
                                                                                          (self.width, self.height,
                                                                                           self.channels))
        frame = np.round(frame)
        frame = np.clip(frame, 0, 255)
        frame = frame.astype('uint8')
        self.out.write(frame)

    def finish(self):
        self.out.release()

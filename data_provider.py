#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

from tensorpack import *
import tensorpack as tp
import cv2
import numpy as np
import os
import glob
import cPickle as pickle
import video

"""
Usage:

python data_provider.py --lmdb mydb2.lmdb
"""


class DumbTriplets(tp.dataflow.RNGDataFlow):
    """Yield frame triplets for phase 0
    """
    def __init__(self, pattern='/graphics/projects/scratch/wieschol/sync-data/*/*_noaudio.low.mp4',
                 samples=10, shuffle=True, offset_positive=10, offset_negative=20, min_positive=5):
        self.files = glob.glob(pattern)
        self.shuffle = shuffle
        self.offset_positive = offset_positive
        self.min_positive = min_positive
        self.offset_negative = offset_negative
        self.samples = samples

    def get_data(self):
        file_idx = list(range(len(self.files)))

        offset = self.rng.randint(self.offset_positive - self.min_positive) + self.min_positive

        if self.shuffle:
            self.rng.shuffle(file_idx)

        for file_id in file_idx:
            video_fn = self.files[file_id]
            vid = video.Reader(video_fn)

            frames = list(range(vid.frames))[:-self.offset_positive]
            self.rng.shuffle(frames)
            frames = frames[:self.samples]

            for f in frames:
                vid.jump(f)
                anchor = vid.read()
                vid.jump(f + offset)
                positive = vid.read()

                neg_frames = list(range(vid.frames))[:f - self.offset_negative] +\
                    list(range(vid.frames))[f + self.offset_negative:]
                if len(len(neg_frames)) < 1:
                    continue
                neg = self.rng.randint(len(neg_frames))
                vid.jump(neg_frames[neg])
                negative = vid.read()

                try:

                    assert anchor.ndim == 3
                    assert positive.ndim == 3
                    assert negative.ndim == 3

                    yield [anchor, positive, negative]
                except:
                    pass


def get_dump_data():
    df = DumbTriplets()

    augmentors = [
        imgaug.Brightness(30, clip=True),
        imgaug.GaussianNoise(sigma=10),
        imgaug.Contrast((0.8, 1.2), clip=False),
        imgaug.Clip(),
        imgaug.Flip(horiz=True),
        imgaug.Flip(vert=True),
        imgaug.ToUint8(),
        imgaug.RandomCrop(224)
    ]
    df = AugmentImageComponents(df, augmentors, copy=False, index=[0, 1, 2])
    return df


class OnlineTriplets(tp.dataflow.RNGDataFlow):
    """Yield frame triplets from phase 1+
    """
    def __init__(self, pattern='*.tour.p', samples=10, shuffle=True):
        self.samples = samples
        self.shuffle = shuffle

        self.files = glob.glob(pattern)

    def size(self):
        return len(self.files) * self.samples

    def get_data(self):
        root_tours = "/graphics/projects/scratch/wieschol/sync-corr/iter2"
        root_videos = "/graphics/projects/scratch/wieschol/sync-data"
        file_idx = list(range(len(self.files)))
        if self.shuffle:
            self.rng.shuffle(file_idx)

        for file_id in file_idx:
            # get pickle file
            tour_pickle = self.files[file_id]
            tour_cost = tour_pickle.replace("tour.p", "cleaned_coarse.jpg")
            tour_cost = cv2.imread(tour_cost)

            # we skip video pairs with too short video length
            if min(tour_cost.shape[:2]) < 1000:
                continue

            pair = tour_pickle.replace(root_tours + "/", "").replace("+.tour.p", "").replace("_3D_", "/3D_")
            tmp = pair.split("+")
            if tmp[0][-1] != tmp[1][-1]:
                continue
            # /graphics/projects/scratch/wieschol/sync-data/2012-07-31/3D_L0004_noaudio.mp4.h
            left = root_videos + "/" + tmp[0] + "_noaudio.low.mp4"
            right = root_videos + "/" + tmp[1] + "_noaudio.low.mp4"

            if os.path.isfile(left) and os.path.isfile(right):
                with open(tour_pickle) as f:
                    global_tour = np.array(pickle.load(f))
                    used_frames = list(range(len(global_tour)))
                    if self.shuffle:
                        self.rng.shuffle(used_frames)
                    used_frames = used_frames[:min(self.samples, len(used_frames))]

                    for dp in self.sample_frames(left, right, used_frames, global_tour):
                        yield dp

    def sample_frames(self, h1, h2, used_frames, tour):

        vidA = video.Reader(h1)
        vidB = video.Reader(h2)

        for k in used_frames:
            a, p = tour[k, ...]
            aa = a * 10
            pp = p * 10

            vidA.jump(aa)
            frameA = vidA.read()
            vidB.jump(pp)
            frameB = vidB.read()

            yield [frameA, frameB]


if __name__ == '__main__':

    df = get_dump_data()
    df.reset_state()

    for a, p, n in df.get_data():
        img = np.hstack([a, p, n])
        cv2.imshow('img', img)
        cv2.waitKey(0)
    # df = OnlineTriplets(pattern='/graphics/projects/scratch/wieschol/sync-corr/iter2/*tour.p')
    # df.reset_state()

    # for pair in df.get_data():
    #     left, right = pair
    #     cv2.imshow('left', left)
    #     cv2.imshow('right', right)
    #     cv2.waitKey(0)

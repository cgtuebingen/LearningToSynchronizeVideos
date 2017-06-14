#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

"""
Example

python embed.py --mp4 3D_L0002_noaudio.low.mp4
                --load /home/wieschol/git/recipes/experimental/imagenet/train_log/resnet/checkpoint
                --skip 10
"""

from tensorpack import *
import numpy as np
import os
import argparse
import video
import h5py
import sync_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp4', help='path to mp4 video', required=True)
    parser.add_argument('--h5', help='path to h5 embedding (Defaults to ".mp4.h5")')
    parser.add_argument('--skip', help='skip n frames', type=int, default=1)
    parser.add_argument('--batch', help='number of frames in one feedforward step', type=int, default=32)
    parser.add_argument('--load', help='path to model checkpoint', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.mp4)

    if not args.h5:
        args.h5 = args.mp4 + '.h5'

    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(args.load),
        model=sync_net.Model(data_format='NCHW', depth=18),
        input_names=['input_frames'],
        output_names=['encoding']))

    vid = video.Reader(args.mp4)
    embeddings = np.zeros((vid.frames // args.skip, 1000))

    offset = 0
    for frames, num in vid.batch_reader(args.batch, args.skip, resize=(224, 224)):
        batch_embedding = pred([frames])[0]

        embeddings[offset:offset + num, ...] = batch_embedding
        offset += num

    with h5py.File(args.h5, 'w') as hf:
        g1 = hf.create_group('group1')
        g1.create_dataset('dataset1', data=embeddings, chunks=(1, 1000), dtype='float32')

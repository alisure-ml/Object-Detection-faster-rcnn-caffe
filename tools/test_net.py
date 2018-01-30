#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int, help='GPU id to use')
    parser.add_argument('--def', dest='prototxt', default="../models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt", type=str, help='prototxt file defining the network')
    parser.add_argument('--net', dest='caffemodel', default="../data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel", type=str, help='model to test')
    parser.add_argument('--cfg', dest='cfg_file', default="../experiments/cfgs/faster_rcnn_alt_opt.yml", type=str, help='optional config file')
    parser.add_argument('--wait', dest='wait', default=True, type=bool, help='wait until net file exists')
    parser.add_argument('--imdb', dest='imdb_name', default='voc_2007_test', type=str, help='dataset to test')
    parser.add_argument('--comp', dest='comp_mode', action='store_true', help='competition mode')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set config keys')
    parser.add_argument('--vis', dest='vis', action='store_true', help='visualize detections')
    parser.add_argument('--num_dets', dest='max_per_image', default=100, type=int, help='max number of detections per image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)

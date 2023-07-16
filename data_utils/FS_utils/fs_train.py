#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import sys
import argparse
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from ute.utils.arg_pars import parser
from data_utils.FS_utils.update_argpars import update
from ute.ute_pipeline import temp_embed, all_actions


if __name__ == '__main__':


    opt = parser.parse_args()
    opt.subaction = 'all'
    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 64
    opt.num_splits = 32
    # opt.epochs = 1000
    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # load an already trained model (stored in the models directory in dataset_root)
    opt.load_model = False
    opt.learn_prototype = True

    # use background noise (e.g. for YTI)
    opt.bg = False
    # granularity level eval or high
    # opt.gr_lev = ''
    # set root
    opt.dataset_root = '/home/sateesh/fs_data'
    if not os.path.exists(opt.exp_root):
        os.mkdir(opt.exp_root)

    opt.tensorboard_dir = os.path.join(opt.exp_root, opt.description)
    os.mkdir(opt.tensorboard_dir)   

    # update log name and absolute paths
    #print("Value of load emb: {}".format(opt.load_embed_feat))
    opt = update(opt)
    #print("Value of load emb: {}".format(opt.load_embed_feat))

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['rgb']
        all_actions(actions, opt)
    else:
        ret_stat = temp_embed(opt)
        for key, val in self.return_stat.items():
            print(key, val)


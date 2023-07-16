#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import sys
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from ute.utils.arg_pars import parser
from data_utils.YTI_utils.update_argpars import update
from ute.ute_pipeline import temp_embed, all_actions
import argparse


if __name__ == '__main__':

    # set root
    opt = parser.parse_args()
    if not os.path.exists(opt.exp_root):
        os.mkdir(opt.exp_root)
    opt.dataset_root = '/home/sateesh/YTI_data'
    # opt.description = "yti_run2_all"
    opt.tensorboard_dir = os.path.join(opt.exp_root, opt.description)
    os.mkdir(opt.tensorboard_dir)

    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 3000
    opt.num_splits = 32

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # do not load model
    opt.load_model = False

    # use background noise (e.g. for YTI)
    opt.bg = True
    #opt.bg_trh = 75

    # update log name and absolute paths
    opt = update(opt)

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
        all_actions(actions, opt = opt)
    else:
        ret_stat = temp_embed(opt = opt)
        print(ret_stat)
        #out_file = open(os.path.join(opt.dataset_root, opt.tensorboard_dir, "final_results.txt"), "w")
        #parse_return_stat(ret_stat, out_file)






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
from data_utils.DA_utils.update_argpars import update
from ute.ute_pipeline import temp_embed, all_actions


if __name__ == '__main__':
    opt = parser.parse_args()
    # set root
    opt.dataset_root = 'path to dataset'

    opt.subaction = 'all'
    # set feature extension and dimensionality
    opt.ext = 'npy'
    opt.feature_dim = 512

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # load an already trained model (stored in the models directory in dataset_root)
    opt.load_model = True 
    opt.loaded_model_name = 'model path'

    # use background noise (e.g. for YTI)
    opt.bg = False
    # granularity level eval or high
    opt.gr_lev = ''

    # update log name and absolute paths
    opt = update(opt)

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['2020']
        all_actions(actions,opt)
    else:
        temp_embed()



#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import sys
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from data_utils.BF_utils.update_argpars import update

from ute.utils.arg_pars import parser
from ute.ute_pipeline import temp_embed, all_actions


if __name__ == '__main__':

    
    opt = parser.parse_args()
    opt.num_splits = 32
    # set root
    opt.dataset_root = '/home/sateesh/bf_data'
    
    
    if not os.path.exists(opt.exp_root):
        os.mkdir(opt.exp_root)

    opt.tensorboard_dir = os.path.join(opt.exp_root, opt.description)
    os.mkdir(opt.tensorboard_dir)   
    # opt.description = "Training model with protos std normal"

    opt.learn_prototype = True
    # set activity
    # ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    # all
    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 64

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # resume training
    opt.resume = False
    # load an already trained model (stored in the models directory in dataset_root)
    opt.load_model = False
    # opt.loaded_model_name = '%s.pth.tar'

    # update log name and absolute paths
    opt = update(opt)

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat',
                   'pancake']
        all_actions(actions, opt)
    else:
        temp_embed(opt)



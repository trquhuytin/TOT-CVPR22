#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'Deceber 2018'

import os.path as ops
import torch

from ute.utils.util_functions import update_opt_str, dir_check
from ute.utils.logging_setup import path_logger


def update(opt):
    opt.data = ops.join(opt.dataset_root, 'features')
    opt.gt = ops.join(opt.dataset_root, 'groundTruth')
    opt.output_dir = ops.join(opt.dataset_root, 'output')
    opt.mapping_dir = ops.join(opt.dataset_root, 'mapping')
    dir_check(opt.output_dir)
    opt.f_norm = False
    if torch.cuda.is_available():
        opt.device = 'cuda'

    opt.embed_dim = 200
    opt.lr_adj = False
    opt.gr_lev = ''
    # if opt.model_name == 'nothing':
    #     opt.load_embed_feat = True

    update_opt_str(opt)

    logger = path_logger(opt)

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))


    return opt
#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import os
import os.path as ops
from ute.utils.util_functions import update_opt_str, dir_check
from ute.utils.logging_setup import path_logger
import torch



def update(opt):

    opt.data = ops.join(opt.dataset_root, 'features')
    opt.gt = ops.join(opt.dataset_root, 'groundTruth')
    opt.output_dir = ops.join(opt.dataset_root, 'output')
    opt.mapping_dir = ops.join(opt.dataset_root, 'mapping')
    dir_check(opt.output_dir)
    opt.f_norm = True
    if torch.cuda.is_available():
        opt.device = 'cuda'

    if opt.global_pipe:
        opt.embed_dim = 30
    else:
        opt.embed_dim = 40

    if not opt.load_model:
        if opt.global_pipe:
            opt.lr = 1e-4
        else:
            opt.lr = 1e-3
        # opt.epochs = 600
        # opt.freeze_iters = 2000
        # #opt.freeze_iters = [800, 800, 800, 800, 800, 800, 800, 800, 800, 2000]

    opt.bg = False  # YTI argument
    opt.gr_lev = ''  # 50Salads argument
    if opt.model_name == 'nothing':
        opt.load_embed_feat = True

    update_opt_str(opt)

    logger = path_logger(opt)

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

    return opt

#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import numpy as np

from ute.utils.logging_setup import logger


class AuxiliaryGMM:
    def __init__(self):
        self.means_ = [0]
        self.covariances_ = [0]

    def score_samples(self, features):
        result = np.ones(features.shape[0]) * (-np.inf)
        return result


class GMM_trh:
    def __init__(self, gmm):
        self._gmm = gmm
        self.trh = np.inf
        self.mean_score = 0
        self.bg_trh_score = []
        if not isinstance(gmm, AuxiliaryGMM):
            self._define_threshold()

    def _define_threshold(self):
        mean = self._gmm.means_[0]
        self.mean_score = self._gmm.score_samples(mean.reshape(1, -1))

    def score_samples(self, features):
        return self._gmm.score_samples(features)

    def append_bg_score(self, score):
        self.bg_trh_score.append(score)

    def update_trh(self, opt, new_bg_trh=None):
        if self.mean_score != 0:
            new_bg_trh = opt.bg_trh if new_bg_trh is None else new_bg_trh
            self.trh = self.mean_score - new_bg_trh
            # self.trh = self.mean_score - 1




#!/usr/bin/env python

"""Creating dataset out of video features for different models.
"""

__all__ = ''
__author__ = 'Anna Kukleva'
__date__ = 'December 2018'


from torch.utils.data import Dataset
import torch
import numpy as np
import random
from ute.utils.logging_setup import logger
from ute.utils.util_functions import join_data
import math

class FeatureDataset(Dataset):
    def __init__(self, videos, features):
        logger.debug('Creating feature dataset')

        self._features = features
        self._gt = None
        # self._videos_features = features
        self._videos = videos

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        gt_item = self._gt[idx]
        features = self._features[idx]
        return np.asarray(features), gt_item

    # @staticmethod
    # def _gt_transform(gt):
    #     if opt.model_name == 'tcn':
    #         return np.array(gt)[..., np.newaxis]
    #     if opt.model_name == 'mlp':
    #         return np.array(gt)

class VideoRelTimeDataset(FeatureDataset):
    def __init__(self, num_frames, num_splits, videos, features, opt):
        logger.debug('Relative time labels')
        super().__init__(videos, features)

        self._video_features_list = []  # used only if opt.concat > 1
        self._video_gt_list = []
        self._action_gt_list = []
        self.num_frames = num_frames
        self.num_splits = num_splits
        self.opt = opt
        

        print("Number of videos: {}".format(len(self._videos)))
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            action_gt = video.gt
            #temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            self._video_features_list.append(video_features)
            self._video_gt_list.append(time_label)
            self._action_gt_list.append(action_gt)
        print("Length of video dataset: {}".format(len(self._video_gt_list)))
        
    def __len__(self):
        return len(self._video_gt_list)

    def __getitem__(self, idx):
        
        gt_item = self._video_gt_list[idx]
        features = self._video_features_list[idx]
        gt_sample, features_sample  = self.uniform_sample(gt_item, features)
        
        return np.asarray(features_sample), gt_sample
    
    def random_sample(self, gt_item, features):
        
        sample_mask = np.sort(random.sample(list(np.arange(features.shape[0])), self.num_frames))
        return gt_item[sample_mask], features[sample_mask]

    def uniform_sample(self, gt_item, features):

        splits = np.arange(self.num_splits) *(math.floor(features.shape[0]/self.num_splits))
        splits = np.repeat(splits, self.num_frames/self.num_splits, axis = 0)
        indices = np.sort(splits + random.choices(list(np.arange(math.floor(features.shape[0]/self.num_splits))), k = self.num_frames))
  
        return gt_item[indices], features[indices]

class VideoRelTimeDataset_tcn(FeatureDataset):
    def __init__(self, num_frames, num_splits, videos, features, opt):
        logger.debug('Relative time labels')
        super().__init__(videos, features)

        self._video_features_list = []  # used only if opt.concat > 1
        self._video_gt_list = []
        self._action_gt_list = []
        self.num_frames = num_frames
        self.num_splits = num_splits
        self.opt = opt
        

        print("Number of videos: {}".format(len(self._videos)))
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            action_gt = video.gt
            #temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            self._video_features_list.append(video_features)
            self._video_gt_list.append(time_label)
            self._action_gt_list.append(action_gt)
        print("Length of video dataset: {}".format(len(self._video_gt_list)))
        
    def __len__(self):
        return len(self._video_gt_list)

    def __getitem__(self, idx):
        
        gt_item = self._video_gt_list[idx]
        features = self._video_features_list[idx]
        gt_sample, features_sample  = self.uniform_sample(gt_item, features)
        
        return np.asarray(features_sample), gt_sample
    
    def random_sample(self, gt_item, features):
        
        sample_mask = np.sort(random.sample(list(np.arange(features.shape[0])), self.num_frames))
        return gt_item[sample_mask], features[sample_mask]

    def uniform_sample(self, gt_item, features):

        splits = np.arange(self.num_splits) *(math.floor(features.shape[0]/self.num_splits))
        splits = np.repeat(splits, (self.num_frames//2)/self.num_splits, axis = 0)
        indices = (splits + random.choices(list(np.arange(math.floor(features.shape[0]/self.num_splits))), k = (self.num_frames//2)))
        indices_positive = indices + np.array(random.choices(list(np.arange(-self.opt.window_size, self.opt.window_size)), k = indices.shape[0]))
        indices = np.sort(np.concatenate([indices, indices_positive]))
        indices = np.clip(indices, 0, features.shape[0] - 1)       
  
        return gt_item[indices], features[indices]





class GTDataset(FeatureDataset):
    def __init__(self, videos, features):
        logger.debug('Ground Truth labels')
        super().__init__(videos, features)

        for video in self._videos:
            gt_item = np.asarray(video.gt).reshape((-1, 1))
            # video_features = self._videos_features[video.global_range]
            # video_features = join_data(None, (gt_item, video_features),
            #                            np.hstack)
            self._gt = join_data(self._gt, gt_item, np.vstack)

            # self._features = join_data(self._features, video_features,
            #                            np.vstack)


class RelTimeDataset(FeatureDataset):
    def __init__(self, videos, features):
        logger.debug('Relative time labels')
        super().__init__(videos, features)

        temp_features = None  # used only if opt.concat > 1
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            # video_features = join_data(None, (time_label, video_features),
            #                             np.hstack)

class TCNDataset(FeatureDataset):
    def __init__(self, videos, features):

        logger.debug('Relative time labels')
        super().__init__(videos, features)

        temp_features = None  # used only if opt.concat > 1
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            
            
            pos_indices = np.arange(len(video_features)) + random.choices(range(1, 30), k = len(video_features))
            pos_indices = np.minimum(pos_indices, len(video_features) - 1)
            video_features_pos = video_features[pos_indices]
            video_features = np.concatenate([np.expand_dims(video_features, axis = 1), np.expand_dims(video_features_pos, axis = 1)], axis = 1)
            
        

            temp_features = join_data(temp_features, video_features, np.vstack)
            #print("Temp features: {}".format(temp_features.shape))

            self._gt = join_data(self._gt, time_label, np.vstack)
        self._features = temp_features

def load_ground_truth(videos, features, shuffle=True):
    logger.debug('load data with ground truth labels for training some embedding')

    dataset = GTDataset(videos, features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)
    return dataloader


def load_reltime(videos, features, opt, mode="train", shuffle=True):
    logger.debug('load data with temporal labels as ground truth')
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    if opt.model_name == 'mlp':
        if mode == "train":
            #print("Num frames: {}".format(opt.batch_size/opt.num_videos))
            
            
            dataset = VideoRelTimeDataset_tcn(num_frames = int(opt.batch_size/opt.num_videos), num_splits = opt.num_splits, videos = videos, features = features, opt = opt)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.num_videos,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)
        else:
            dataset = RelTimeDataset(videos, features)
    if opt.model_name == 'tcn':
        dataset = TCNDataset(videos, features)

    if mode == "test":
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)

    return dataloader

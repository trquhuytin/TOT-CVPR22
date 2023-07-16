#!/usr/bin/env python

"""Baseline for relative time embedding: learn regression model in terms of
relative time.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import torch
import torch.nn as nn
from torch.nn import functional as F
from ute.utils.logging_setup import logger

class TCN(nn.Module):

    def __init__(self, num_positives, temperature = 0.1):
        super(TCN, self).__init__()
        self.temperature = temperature
        self.num_positives = num_positives

        
    def _npairs_loss(self, labels, embeddings_anchor, embeddings_positive):
        """Returns n-pairs metric loss."""
        
        # Get per pair similarities.
        similarity_matrix = torch.matmul(
            embeddings_anchor, embeddings_positive.t())

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = labels.shape

        # Add the softmax loss.
        xent_loss = F.cross_entropy(
            input=similarity_matrix, target=labels)
        #xent_loss = tf.reduce_mean(xent_loss)

        return xent_loss


    def single_sequence_loss(self, embs):
        """Returns n-pairs loss for a single sequence."""

        labels = torch.arange(embs.shape[0]//self.num_positives).long().cuda()
        embs = embs/self.temperature #temperature to scale up the embeddings
        indices = torch.arange(0, embs.shape[0], step=self.num_positives) + torch.randint(0, self.num_positives - 1, size = (labels.shape[0],))
        indices_positive = indices + 1
        embeddings_anchor = embs[indices]
        embeddings_positive = embs[indices_positive]
        loss = self._npairs_loss(labels, embeddings_anchor, embeddings_positive)
        return loss

    def forward(self, embs):
        return self.single_sequence_loss(embs)



class ClusterLoss(nn.Module):

    def __init__(self, num_clusters, alpha = 0.01):
      super(ClusterLoss, self).__init__()
      self.num_clusters = num_clusters
      self.alpha = alpha

    def cluster_sim_loss(self, prototypes):

      sim = torch.matmul(prototypes, torch.transpose(prototypes, 0 , 1)) # N x N
      
      p_norm = torch.norm(prototypes, dim = 1).unsqueeze(1) # N x 1
      
      p_norm_matrix = torch.matmul(p_norm, torch.transpose(p_norm, 0, 1)) # N x N
      cosine_sim = sim / p_norm_matrix
      return cosine_sim

    def entropy_loss(self, prob, epsilon = 1e-7):
      return -torch.sum(prob*(torch.log(prob + epsilon)))



    def cosine_sim(self, embeddings, prototypes):

      sim = torch.matmul(embeddings, torch.transpose(prototypes, 0, 1))
      emb_norm = torch.norm(embeddings, dim = 1).unsqueeze(1)
      proto_norm = torch.norm(prototypes, dim  = 1).unsqueeze(1)
      norm_matrix = torch.matmul(emb_norm, torch.transpose(proto_norm, 0, 1))
      cosine_sim = sim/norm_matrix
      
      return cosine_sim

    def cluster_diversity_loss(self, batch_size, cluster_logits):

      cluster_sums = torch.sum(cluster_logits, axis = 1)
      cluster_sums = cluster_sums - batch_size/self.num_clusters
      cluster_sums = torch.max(cluster_sums, torch.tensor([0.]).cuda())
      return torch.sum(cluster_sums)

    def cluster_entropy_loss(self, probs):
    
      marginal_probs = torch.sum(probs, dim = 0).float()
      marginal_probs = torch.div(marginal_probs, probs.shape[0])
      entropy = self.entropy_loss(marginal_probs)
      return entropy
       

    def prototype_loss(self, embeddings, prototypes):

      dists = self.compute_euclidean_dist(embeddings, prototypes)
      sim = 1/dists # dist -> similarity
      sim_soft = F.softmax(sim, dim = 1) #softmax normalization
      cluster_assignments =  F.one_hot(torch.argmax(sim_soft, dim = 1), num_classes = self.num_clusters)
      cluster_assignments.requires_grad = False
      entropy = self.cluster_entropy_loss(sim_soft)
      inter_class_variances = torch.sum(cluster_assignments * (dists))
     
      return inter_class_variances, entropy, intra_class_variances

    def forward(self, embeddings, prototypes):
      return self.prototype_loss(embeddings, prototypes)


    def compute_euclidean_dist(self,  embeddings, prototypes):
      dists = torch.sum(embeddings**2, dim = 1).view(-1, 1) + torch.sum(prototypes**2, dim = 1).view(1, -1) -2 * torch.matmul(embeddings, torch.transpose(prototypes, 0, 1)) 
      #print(dists.shape)
      return dists

class MLP_bigger(nn.Module):

    def __init__(self, opt, num_clusters):
        super(MLP_bigger, self).__init__()

        ### concat
        self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim * 2)
        self.fc3 = nn.Linear(opt.embed_dim * 2, opt.embed_dim * 2)
        self.fc4 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.prototype_layer = nn.Linear(opt.embed_dim, num_clusters, bias=False)
        #self.fc_last = nn.Linear(opt.embed_dim, 1)
        self._init_weights()

    def forward(self, x):
        shape_dims = [*x.shape]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        proto_scores = self.prototype_layer(x)
        #x = self.fc_last(x)
        
        return x, proto_scores, x

    def embedded(self, x):

        shape_dims = [*x.shape]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x

    def get_prototypes(self):

        return self.prototype_layer.weight.data.clone()

    def _init_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if type(m.bias) != type(None):
                  nn.init.constant_(m.bias, 0)

class MLP(nn.Module):

    def __init__(self, opt, num_clusters):
        super(MLP, self).__init__()

        ### concat
        self.opt = opt
        self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        
        # self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim )
       
        self.prototype_layer = nn.Linear(opt.embed_dim, num_clusters, bias=False)
        if opt.time_loss:
          self.fc_last = nn.Linear(opt.embed_dim, 1)
        
        self._init_weights()

    def forward(self, x):
        shape_dims = [*x.shape]

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        embs = x.clone()
        proto_scores = self.prototype_layer(x)
        
        if self.opt.time_loss:
          x = self.fc_last(x)
        
        return x, proto_scores, embs

    def embedded(self, x):

        shape_dims = [*x.shape]
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

    def get_prototypes(self):

        return self.prototype_layer.weight.data.clone()

    def _init_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if type(m.bias) != type(None):
                  nn.init.constant_(m.bias, 0)


def create_model(opt, num_clusters, learn_prototype = True):

    
    torch.manual_seed(opt.seed)
   
    if opt.model == "mlp_b":
      print("BIGGER MODEL INITIALIZED")
      model = MLP_bigger(num_clusters = num_clusters, opt = opt).to(opt.device)
    else:
      model = MLP(num_clusters = num_clusters, opt = opt).to(opt.device)
    loss = nn.MSELoss(reduction='sum')
    tcn_loss = TCN(int((opt.batch_size/opt.num_videos)/opt.num_splits))
    print(int((opt.batch_size/opt.num_videos)/opt.num_splits))
    
    # loss = nn.MSELoss().cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params,
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))

    return model, loss, tcn_loss, optimizer


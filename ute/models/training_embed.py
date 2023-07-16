#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import io
from imageio import imread
from ute.utils.logging_setup import logger
from ute.utils.util_functions import Averaging, adjust_lr
from ute.utils.util_functions import dir_check, save_params
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def training(train_loader, epochs, save, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    
    logger.debug('create model')
    
    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    learn_prototype = kwargs['learn_prototype']
    tcn_loss = kwargs['tcn_loss']
    opt = kwargs['opt']
    
    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(opt.tensorboard_dir)
    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()
    c_losses = Averaging()
    tcn_losses = Averaging()

    adjustable_lr = opt.lr

    logger.debug('epochs: %s', epochs)
    f = open("test_q_distribution.npy", "wb")
    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        logger.debug('Epoch # %d' % epoch)
      
        end = time.time()
      
        for i, (features, labels) in enumerate(train_loader):

            num_videos = features.shape[0]
            features = torch.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
            labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], labels.shape[2]))
           
          
            data_time.update(time.time() - end)
            features = features.float()
            labels = labels.float().to(opt.device)
            if opt.device == 'cuda':
                features = features.cuda(non_blocking=True)

            output, proto_scores, embs = model(features)
        
            if learn_prototype:
                with torch.no_grad():

                    #compute q
                    p_gauss = get_cost_matrix(batch_size = opt.batch_size, num_videos = num_videos, num_videos_dataset = opt.num_videos \
                    ,sigma = opt.sigma, num_clusters = proto_scores.shape[1])

                    if opt.apply_temporal_ot:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, p_gauss)
                    else:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, None)
                    if (i + (epoch * len(train_loader))) % 500 == 0:
                        img_finalq= plot_confusion_matrix(q.clone().detach().cpu().numpy())
                        img_protos = plot_confusion_matrix(proto_scores.clone().detach().cpu().numpy())
                        prototypes = model.get_prototypes()
                        dists = compute_euclidean_dist(prototypes, prototypes)
                        img_dists = plot_confusion_matrix(dists.detach().cpu().numpy())

                        writer.add_image("Q Matrix", img_finalq, i + (epoch * len(train_loader)))
                        writer.add_image("Dot-Product Matrix", img_protos,  i + (epoch * len(train_loader)))
                        writer.add_image("Prototype Dists", img_dists, i + (epoch * len(train_loader)))
                        #np.save(f, q.clone().detach().cpu().numpy())
            
                proto_probs = F.softmax(proto_scores/opt.temperature)
                 
                if i + (epoch * len(train_loader)) % 500 == 0:
                    with torch.no_grad():
                        img = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy())
                        writer.add_image("P Matrix", img, i + (epoch * len(train_loader)))

                proto_probs = torch.clamp(proto_probs, min= 1e-30, max=1)
                proto_loss = torch.mean(torch.sum(q*torch.log(proto_probs), dim = 1))
            
            loss_tcn =  tcn_loss(embs)
            loss_values = 0 
            if opt.tcn_loss:
                loss_values += loss_tcn 
            
            if opt.time_loss:
                loss_values = loss(output.squeeze(), labels.squeeze())  #+ loss_tcn(embeddings) #loss_tcn(embeddings) 
            
            if learn_prototype and (i + (epoch * len(train_loader))) >= opt.freeze_proto_loss:

               loss_values -= proto_loss
               
            c_losses.update(proto_loss.item(), 1)
            tcn_losses.update(loss_tcn.item(), 1)

            losses.update(loss_values.item(), features.size(0))
            

            optimizer.zero_grad()
            loss_values.backward()
            if i + (epoch * len(train_loader)) < opt.freeze_iters:

                for name, p in model.named_parameters():
                   
                    if "prototype" in name:
                        p.grad = None 

            
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            writer.add_scalar("Loss/task_loss", loss_values, i + (epoch * len(train_loader)))
            if learn_prototype:
                writer.add_scalar("Loss/cluster_loss", -proto_loss, i + (epoch * len(train_loader)))
                writer.add_scalar("Loss/tcn_loss", loss_tcn, i + (epoch * len(train_loader)))

            if i % 20 == 0 and i:
                
                
                if not learn_prototype:
                    #print("HERE")
                
                    logger.debug('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
                else:
                    print("HEREEEEE")
                    logger.debug('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Cluster Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
                                'TCN Loss {tcn_loss.val: .4f} ({tcn_loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, c_loss = c_losses, tcn_loss = tcn_losses))

        logger.debug('loss: %f' % losses.avg)
     
        losses.reset()
    
    f.close()

    opt.resume_str = join(opt.tensorboard_dir, 'models',
                          '%s.pth.tar' % opt.log_str)
    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        if opt.global_pipe:
            dir_check(join(opt.dataset_root, 'models', 'global'))
            opt.resume_str = join( opt.tensorboard_dir, 'models', 'global',
                                  '%s.pth.tar' % opt.log_str)
        else:
            dir_check(join(opt.tensorboard_dir, 'models'))
            save_params(opt, join(opt.tensorboard_dir))
        torch.save(save_dict, opt.resume_str)
    return model

def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        curr_sum = torch.sum(Q, dim=1)
        # dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            # dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def load_model():
    if opt.loaded_model_name:
        if opt.global_pipe:
            resume_str = opt.loaded_model_name
        else:
            resume_str = opt.loaded_model_name #% opt.subaction
        # resume_str = opt.resume_str
    else:
        resume_str = opt.log_str + '.pth.tar'
    # opt.loaded_model_name = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(join(opt.dataset_root, 'models',
                                     '%s' % resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(join(opt.tensorboard_dir, 'models',
                                 '%s' % resume_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s' % resume_str)
    return checkpoint


def get_cost_matrix(batch_size, num_videos, num_videos_dataset, num_clusters, sigma):

    cost_matrix = generate_matrix(int(batch_size/num_videos_dataset), num_clusters)
    cost_matrix = np.vstack([cost_matrix] * num_videos)
    p_gauss = gaussian(cost_matrix, sigma = sigma)

    return p_gauss


def cost(i, j, n, m):

  return ((i - (j/m) *n)/n)**2


def cost_paper(i, j, n, m):
    
    return ((abs(i/n - j/m))/(np.sqrt((1/n**2) + (1/m**2))))**2


def gaussian(cost, sigma):

    #print("Value of sigma: {}".format(opt.sigma))
    return (1/(sigma * 2*3.142))*(np.exp(-cost/(2*(sigma**2))))

def generate_matrix(num_elements, num_clusters):

    cost_matrix = np.zeros((num_elements, num_clusters))

    for i in range(num_elements):
        for j in range(num_clusters):

            cost_matrix[i][j] = cost_paper(i, j, num_elements, num_clusters)

    return cost_matrix

def plot_confusion_matrix(q):

    fig, ax = plt.subplots(nrows=1)
    sns.heatmap(q, ax = ax)
    image = plot_to_image(fig)

    return image

def compute_euclidean_dist(embeddings, prototypes):

    with torch.no_grad():

        dists = torch.sum(embeddings**2, dim = 1).view(-1, 1) + torch.sum(prototypes**2, dim = 1).view(1, -1) -2 * torch.matmul(embeddings, torch.transpose(prototypes, 0, 1)) 
        return dists

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = imread(buf)

  # Add the batch dimension
  #image = tf.expand_dims(image, 0)
  return image.transpose(2, 0, 1)


def generate_optimal_transport_labels(proto_scores, epsilon, p_gauss):
    
    q = proto_scores/epsilon
    q =  torch.exp(q)
    if p_gauss is not None:
        q = q * torch.from_numpy(p_gauss).cuda()
    q = q.t()
    q =  distributed_sinkhorn(q, 3)

    return q

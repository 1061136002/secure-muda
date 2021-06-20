import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def loss_CE(M_logits, cls_labels):
    return nn.CrossEntropyLoss(reduction='mean')(M_logits,cls_labels)

def l4_mirror_CE(M_logits,cls_labels):
    #by default matrix is batch x domain x class

    cls_labels = cls_labels.view(-1,1)
    n_batch,n_domain,n_class = M_logits.shape
    cls_M_logits  = M_logits.permute(0,2,1)  #batch x class x domain
    cls_labels         = cls_labels.expand(n_batch,n_domain)
    return nn.CrossEntropyLoss(reduction='mean')(cls_M_logits,cls_labels)


######################## Wasserstein loss ########################

# The wasserstein code is as implemented in https://github.com/eifuentes/swae-pytorch

def rand_projections(embedding_dim, num_projections=50):
    """This function generates `num_projections` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_projections (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_projections, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_projections, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection


    # return torch.sort(wasserstein_distance.mean(dim=-1))[0][:num_projections // 10].mean()
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples,
                                gaussian_samples,
                                num_projections=50,
                                p=2,
                                device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, gaussian_samples,
                                       num_projections, p, device)
    return swd


############################################################------------------_SECTION METRIC--------------------###########################################################################

def acc_metric(preds,labels):
    return np.mean(np.asarray(preds) == np.asarray(labels))

def get_metric(key, feats):
    if key == 'cls_acc':
        cls_preds        = feats['cls_preds']
        cls_labels       = feats['cls_labels']
        return acc_metric(cls_preds,cls_labels)

    elif key == 'cls_acc_data':
        cls_preds        = np.array(feats['all_preds'])
        cls_labels       = np.array(feats['all_labels'])
        metric           = np.sum((cls_preds == cls_labels).astype(float))
        return metric,len(cls_labels)

    return metric

############################################################------------------_SECTION LOGITS-------------------###########################################################################

def get_logits(feats):
    M = feats['M']

    cls_logits = M.softmax(dim=-1)
    return cls_logits,None,M

import numpy as np

import metrics as metrics

from tqdm import tqdm

import torch
import time

def sample_from_gaussians(means, covs, n_samples, perm_res=True):
    # Return samples from the num_classes gaussians trained on the source 
    N_CLASSES = len(n_samples)

    Xs = []
    Ys = []
    
    for i in range(N_CLASSES):
        if n_samples[i] > 0:
            curr_x = np.random.multivariate_normal(means[i], covs[i], n_samples[i])
            curr_y = np.repeat(i, n_samples[i])
                        
            Xs.append(curr_x)
            Ys.append(curr_y)

    Xs = np.vstack(Xs)
    Ys = np.concatenate(Ys)

    if not perm_res:
        return Xs, Ys
    else:
        perm = np.random.permutation(Xs.shape[0])
        return Xs[perm,:], Ys[perm]

def learn_gaussians(trainer_S, debug=False):
    ###########################################################################################################################
    # Extract source domain latent features from all sources
    ###########################################################################################################################s
    trainer_S.set_mode(trainer_S.settings['mode']['train']) # keep the distribution as during source training
    trainer_S.get_all_train_src_dataloaders()

    with torch.no_grad():
        # Gather samples from both source domains
        all_labels_src                  =  []
        all_preds_src                   =  []
        all_confs_src                   =  []
        all_F_src                       =  []
        all_M_src                       =  []

        dom = trainer_S.src_domain

        source_dl_iter_train_list = trainer_S.source_dl_iter_train_list

        for data in tqdm(source_dl_iter_train_list, desc=dom):
            indx,images,label,_         = data
            x                           = images.to(trainer_S.settings['device']).float()
            label                       = label.to(trainer_S.settings['device']).long()
            F                           = trainer_S.network.model['global']['Fs'](x)
            M                           = trainer_S.network.model['global']['M'](F)

            cls_logits,_,mat            = metrics.get_logits(feats={'M':M})
            cls_confs,cls_preds         = torch.max(cls_logits,dim=-1)

            all_labels_src.extend(list(label.cpu().numpy()))
            all_preds_src.extend(list(cls_preds.cpu().numpy()))
            all_confs_src.extend(list(cls_confs.cpu().numpy()))
            all_F_src.append(F)
            all_M_src.append(M)

        all_labels_src = np.asarray(all_labels_src)
        all_preds_src = np.asarray(all_preds_src)
        all_confs_src = np.asarray(all_confs_src)
        all_F_src = torch.cat(all_F_src,dim=0).cpu().numpy()
        all_M_src = torch.cat(all_M_src,dim=0).cpu().numpy()

    ###########################################################################################################################
    # Learn means and covariances
    ###########################################################################################################################
    adapt_lvl = all_F_src

    N_CLASSES = trainer_S.settings['num_C'][trainer_S.src_domain]
    Z_SIZE = adapt_lvl.shape[-1]
    TAU = trainer_S.settings['tau'] * len(trainer_S.settings['src_datasets'])

    trainer_S.means = np.zeros((N_CLASSES, Z_SIZE))
    trainer_S.covs = np.zeros((N_CLASSES, Z_SIZE, Z_SIZE))

    for c in range(N_CLASSES):
        idx = (all_labels_src == c) & (all_preds_src == c) & (all_confs_src > TAU)

        trainer_S.means[c] = np.mean(adapt_lvl[idx], axis=0)
        trainer_S.covs[c] = np.dot((adapt_lvl[idx] - trainer_S.means[c]).T, (adapt_lvl[idx] - trainer_S.means[c])) / np.sum(idx)
        

    if debug == True:
        # Store intermediate results for debug purposes
        trainer_S.adapt_lvl = adapt_lvl
        trainer_S.all_labels_src = all_labels_src

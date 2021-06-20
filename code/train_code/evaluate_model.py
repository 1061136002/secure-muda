import numpy as np

import config as config
import metrics as metrics
import gaussian_utils

from trainer import MultiSourceTrainer

from tqdm import tqdm

import torch
from torch.autograd import Variable

def get_individual_performance():
    # Computes the target performance for source-only and post-adaptation models
    # First outputs source only performance for each source domain, followed by target performance for each source domain

    pre_adapt_accs = []
    post_adapt_accs = []
    
    # First, see the initial performance of the models
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        trainer_S.load_model_weights(it_thresh='enough_iter')

        pre_adapt_accs.append(trainer_S.val_over_target_set(save_weights=False))
        
    # Next, the performance after adaptation
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        trainer_S.load_model_weights(it_thresh='max_iter')

        post_adapt_accs.append(trainer_S.val_over_target_set(save_weights=False))
    
    return np.asarray(pre_adapt_accs), np.asarray(post_adapt_accs)

def get_target_accuracy(weights, it_thresh='max_iter'):
    # Computes the accuracy obtained by combining logits from several models

    logit_sum = None

    all_logits_dict = {}
    all_labels_dict = {}

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        trainer_S = MultiSourceTrainer(src_domain_idx)
        # trainer_S.load_model_weights('model_' + it_thresh + str(trainer_S.settings[it_thresh]) + '.pth')
        trainer_S.load_model_weights(it_thresh=it_thresh)
        
        trainer_S.set_mode(trainer_S.settings['mode']['val'])

        with torch.no_grad():
            # Gather samples from both source domains
            all_labels_tar                  =  []
            all_preds_tar                   =  []
            all_F_src                       =  []
            all_logits                      =  []

            dom = trainer_S.trgt_domain

            trainer_S.get_all_val_target_dataloaders()
            target_dl_iter_val_list = trainer_S.target_dl_iter_val_list

            for data in tqdm(target_dl_iter_val_list, desc=dom):
                indx,images,label,_         = data
                x                           = images.to(trainer_S.settings['device']).float()
                label                       = label.to(trainer_S.settings['device']).long()
                F                           = trainer_S.network.model['global']['Fs'](x)
                M                           = trainer_S.network.model['global']['M'](F)

                cls_logits,_,mat            = metrics.get_logits(feats={'M':M})
                all_logits.append(cls_logits)

                all_labels_tar.extend(list(label.cpu().numpy()))

            all_labels_tar = np.asarray(all_labels_tar)
            all_logits = torch.cat(all_logits, dim=0).cpu().numpy()

            all_logits_dict[src_domain_idx] = np.copy(all_logits)
            all_labels_dict[src_domain_idx] = np.copy(all_labels_tar)

            if logit_sum is None:
                logit_sum = weights[src_domain_idx] * all_logits
            else:
                logit_sum += weights[src_domain_idx] * all_logits
    
    labels_hat = np.argmax(logit_sum, axis=-1)
    
    return np.mean(all_labels_tar == labels_hat), labels_hat, all_labels_tar

# Learn the weights w minimizing the generalizability objective
def learn_w_generalizability(it_thresh='max_iter', num_steps=300):
    trainers = {}
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        dom = config.settings['src_datasets'][src_domain_idx]
        trainers[dom] = MultiSourceTrainer(src_domain_idx)
        trainers[dom].load_model_weights(it_thresh=it_thresh)
        trainers[dom].set_mode(trainers[dom].settings['mode']['val'])

        trainers[dom].src_data = {}
        trainers[dom].initialize_src_train_dataloader()

    w = torch.ones(len(config.settings['src_datasets']))
    w.requires_grad = True
    w_optim = torch.optim.Adam([w], lr=1e-2)

    for step in range(num_steps):
        for dom in config.settings['src_datasets']:
            try:
                trainers[dom].src_data[dom]={}
                _,trainers[dom].src_data[dom]['images'],trainers[dom].src_data[dom]['label'],trainers[dom].src_data[dom]['domain_label'] = trainers[dom].source_dl_iter_train_list.next()
                trainers[dom].src_data[dom]['images'] = Variable(trainers[dom].src_data[dom]['images']).to(trainers[dom].settings['device']).float()
                trainers[dom].src_data[dom]['label'] = Variable(trainers[dom].src_data[dom]['label']).to(trainers[dom].settings['device']).long()
                trainers[dom].src_data[dom]['domain_label'] = Variable(trainers[dom].src_data[dom]['domain_label']).to(trainers[dom].settings['device']).long()
            except StopIteration:
                trainers[dom].initialize_src_train_dataloader()
                trainers[dom].src_data[dom]={}
                _,trainers[dom].src_data[dom]['images'],trainers[dom].src_data[dom]['label'],trainers[dom].src_data[dom]['domain_label'] = trainers[dom].source_dl_iter_train_list.next()
                trainers[dom].src_data[dom]['images'] = Variable(trainers[dom].src_data[dom]['images']).to(trainers[dom].settings['device']).float()
                trainers[dom].src_data[dom]['label'] = Variable(trainers[dom].src_data[dom]['label']).to(trainers[dom].settings['device']).long()
                trainers[dom].src_data[dom]['domain_label'] = Variable(trainers[dom].src_data[dom]['domain_label']).to(trainers[dom].settings['device']).long()

            logit_sum = torch.zeros((trainers[dom].src_data[dom]['images'].shape[0], trainers[dom].N_CLASSES)).to(trainers[dom].settings['device'])
            for eval_domain_idx in range(len(config.settings['src_datasets'])):
                eval_dom = config.settings['src_datasets'][eval_domain_idx]

                if eval_dom == dom:
                    continue

                with torch.no_grad():
                    F                           = trainers[eval_dom].network.model['global']['Fs'](trainers[dom].src_data[dom]['images'])
                    M                           = trainers[eval_dom].network.model['global']['M'](F)
                logit_sum += M * w[eval_domain_idx]

            loss = metrics.loss_CE(logit_sum, trainers[dom].src_data[dom]['label']) + torch.abs(1 - torch.sum(w))

            w_optim.zero_grad()
            loss.backward()
            w_optim.step()

        if step % (num_steps // 10) == 0:
            print(step, loss.item(), w)
        
    return w.detach().numpy()

# Learn the weights w minimizing the wasserstein objective
def learn_w_w2(it_thresh='max_iter', num_steps=100):
    w2_dist = np.zeros(len(config.settings['src_datasets']))
    
    # The approximation is better if the batch size is larger. Let's see how large we can make it
    multiplier = 5
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]
        trainer = MultiSourceTrainer(src_domain_idx)
        trainer.load_model_weights(it_thresh=it_thresh)
        trainer.set_mode(trainer.settings['mode']['val'])
        initial_batch_size = trainer.adapt_batch_size
        
        for curr_multiplier in range(1, 6):
            print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

            # align the batch sizes
            trainer.adapt_batch_size = initial_batch_size * curr_multiplier
            trainer.batch_size = initial_batch_size * curr_multiplier

            if trainer.batch_size > 600:
                multiplier = min(multiplier, curr_multiplier - 1)
                break

            try:
                trainer.initialize_src_train_dataloader()
                _,X_src,_,_ = trainer.source_dl_iter_train_list.next()
                X_src = Variable(X_src).to(trainer.settings['device']).float()

                trainer.initialize_target_adapt_dataloader()
                _,X_tar,_,_ = trainer.adapt_target_dl_iter_train_list.next()
                X_tar = Variable(X_tar).to(trainer.settings['device']).float()
            except StopIteration:
                print("Multiplier for {} is {}".format(dom, curr_multiplier - 1))
                multiplier = min(multiplier, curr_multiplier - 1)
                break
                
    print('multiplier = {}'.format(multiplier))
    print("\n\n")
    
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        print("Loading trainer for source domain {}".format(config.settings['src_datasets'][src_domain_idx]))

        dom = config.settings['src_datasets'][src_domain_idx]
        trainer = MultiSourceTrainer(src_domain_idx)
        
        # learn the gaussians
        trainer.load_model_weights(it_thresh='enough_iter')
        gaussian_utils.learn_gaussians(trainer)
        n_samples = np.ones(trainer.N_CLASSES, dtype=int) * trainer.settings["gaussian_samples_per_class"] * 100
        trainer.gaussian_z, trainer.gaussian_y = gaussian_utils.sample_from_gaussians(trainer.means, trainer.covs, n_samples)
        
        trainer.pseudo_target_dist = np.ones(trainer.N_CLASSES, dtype=int)
            
        print("Dist=")
        print(trainer.pseudo_target_dist)
        
        trainer.load_model_weights(it_thresh=it_thresh)
        trainer.set_mode(trainer.settings['mode']['val'])
        
        trainer.adapt_batch_size *= multiplier
        trainer.batch_size = trainer.adapt_batch_size    # align the batch sizes

        trainer.initialize_src_train_dataloader()
        trainer.initialize_target_adapt_dataloader()
        
        # Get the mean W2 distance between encodings of the current domain and the target domain
        # Get a source sample
        w2_dist[src_domain_idx] = 0
        for step in range(num_steps):
            # Get a source sample
            try:
                _,X_src,_,_ = trainer.source_dl_iter_train_list.next()
                X_src = Variable(X_src).to(trainer.settings['device']).float()
            except StopIteration:
                trainer.initialize_src_train_dataloader()
                _,X_src,_,_ = trainer.source_dl_iter_train_list.next()
                X_src = Variable(X_src).to(trainer.settings['device']).float()

            # Get a target sample
            try:
                _,X_tar,_,_ = trainer.adapt_target_dl_iter_train_list.next()
                X_tar = Variable(X_tar).to(trainer.settings['device']).float()
            except StopIteration:
                trainer.initialize_target_adapt_dataloader()
                _,X_tar,_,_ = trainer.adapt_target_dl_iter_train_list.next()
                X_tar = Variable(X_tar).to(trainer.settings['device']).float()
                
            # Compute the number of gaussian samples to be used for the current batch
            normalized_dist = trainer.pseudo_target_dist / np.sum(trainer.pseudo_target_dist)
            num_samples = np.array(normalized_dist * trainer.adapt_batch_size, dtype=int)
            while trainer.adapt_batch_size > np.sum(num_samples):
                idx = np.random.choice(range(trainer.N_CLASSES), p = normalized_dist)
                num_samples[idx] += 1

            # Get gaussian samples for the current batch
            gz = []
            gy = []
            for c in range(trainer.N_CLASSES):
                ind = np.where(trainer.gaussian_y == c)[0]
                ind = ind[np.random.choice(range(len(ind)), num_samples[c], replace=False)]
                gz.append(trainer.gaussian_z[ind])
                gy.append(trainer.gaussian_y[ind])
            gz = np.vstack(gz)
            gy = np.concatenate(gy)

            gz = torch.as_tensor(gz).to(trainer.settings['device']).float()
            gy = torch.as_tensor(gy).to(trainer.settings['device']).long()

            with torch.no_grad():
                f_src = trainer.network.model['global']['Fs'](X_src)
                f_tar = trainer.network.model['global']['Fs'](X_tar)
                
                d1 = metrics.sliced_wasserstein_distance(f_src, gz, trainer.settings['num_projections'], 2, trainer.settings['device']).item()
                d2 = metrics.sliced_wasserstein_distance(f_tar, gz, trainer.settings['num_projections'], 2, trainer.settings['device']).item()
                w2_dist[src_domain_idx] += d1 + d2
    

            if step % (num_steps // 10) == 0:
                print(step, w2_dist, dom)
        
    w = 1 / w2_dist
    w = w / np.sum(w)
        
    return w


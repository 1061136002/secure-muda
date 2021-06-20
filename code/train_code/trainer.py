import numpy as np
import copy
import shutil
import os

import config as config
import metrics as metrics
from net import SingleSourceNet as smodel
from dataset import FeatureDataset
import gaussian_utils

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import time

class MultiSourceTrainer(object):
    def __init__(self, src_domain_idx):
        self.settings                                   = copy.deepcopy(config.settings)
        
        self.src_domain                                 = self.settings['src_datasets'][src_domain_idx]
        self.trgt_domain                                = self.settings['trgt_datasets'][0]
        self.N_CLASSES                                  = self.settings["num_C"][self.src_domain]

        self.network                                    = smodel().to(self.settings['device'])

        self.to_train                                   = self.settings['to_train']

        #batch size
        self.batch_size                                 = self.settings['batch_size']
        self.val_batch_size                             = self.settings['val_batch_size_factor']*self.settings['batch_size']
        self.adapt_batch_size                           = self.settings['adapt_batch_size']
        
        self.current_iteration                          = self.settings['start_iter']
        self.exp_name                                   = self.settings['exp_name']
        self.phase                                      = self.settings['mode']['train']

        #data loader dictionaries
        self.source_dl_iter_train_list                  = []
        self.target_dl_iter_val_list                    = []
        self.adapt_target_dl_iter_train_list            = []

        self.itt_delete                                 = []
        self.max_test_acc                               = -1
        
        self.get_all_train_src_dataloaders()
        self.init_optimizers()

        all_losses                                      = self.optimizer_dict.keys()
        self.active_losses                              = [current_loss for current_loss in all_losses if  self.settings['use_loss'][current_loss]]
        
        self.source_loss_history                        = []
        self.adaptation_loss_history                    = []
        self.target_acc_history                         = []
        self.it_history                                 = []
        
        assert self.settings['enough_iter'] % self.settings['val_after'] == 0
        assert self.settings['max_iter'] % self.settings['val_after'] == 0
        
    def init_folder_paths(self):
        for name in ['weights_path', 'summaries_path']:
            if not os.path.exists(self.settings[name]):
                os.mkdir(self.settings[name])
            if not os.path.exists(os.path.join(self.settings[name],self.settings['exp_name'])):
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name']))
            if not os.path.exists(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain)):
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
            else:
                shutil.rmtree(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
                os.mkdir(os.path.join(self.settings[name],self.settings['exp_name'], self.src_domain))
    
    '''
    Function to load model weights
    '''
    def load_model_weights(self, it_thresh='enough_iter', weights_file=None):
        if weights_file == None:
            weights_file = 'model_' + it_thresh + str(self.settings[it_thresh]) + '.pth'
        load_weights_path = os.path.join(self.settings['weights_path'],self.settings['exp_name'], self.src_domain, weights_file)
        
        dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
        model_state_dict = dict_to_load['model_state_dict']

        for nc,compts in self.network.model.items():
            for name,comp in compts.items():
                self.network.model[nc][name].load_state_dict(model_state_dict['_'.join([nc,name])])

    '''
    Function to load optimizer
    '''
    def load_optimizers(self, opt_file=None):
        if opt_file == None:
            opt_file = 'opt_enough_iter' + str(self.settings['enough_iter']) + '.pth'
        load_weights_path = os.path.join(self.settings['weights_path'], self.settings['exp_name'], self.src_domain, opt_file)
        dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
        optimizer_state_dict = dict_to_load['optimizer_state_dict']

        for name,optimizer in self.optimizer_dict.items():
            if self.settings['use_loss'][name]:
                optimizer.load_state_dict(optimizer_state_dict[name])
    
    def check_and_save_weights(self,curr_cls_acc,dom):
        if self.current_iteration in [self.settings['enough_iter'], self.settings['max_iter']]:
            self.save_weights()
            
    '''
    Function to save model and optimizer state
    '''
    def save_weights(self):
        weights_path = self.settings['weights_path']

        model_state_dict={}
        for nc,compts in self.network.model.items():
            for name,comp in compts.items():
                model_state_dict['_'.join([nc,name])]=comp.cpu().state_dict()

        optimizer_state_dict ={}
        for name,optimizer in self.optimizer_dict.items():
            optimizer_state_dict[name]=optimizer.state_dict()
            
            
        save_dict    = {
                         'model_state_dict':model_state_dict,
                       }
        save_path = os.path.join(self.settings['weights_path'], self.exp_name, self.src_domain)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for it_thresh in ['enough_iter', 'max_iter']:
            if self.current_iteration == self.settings[it_thresh]:
                torch.save(save_dict, os.path.join(save_path, 'model_' + it_thresh + str(self.current_iteration) + '.pth'))
            
        torch.save(save_dict, os.path.join(save_path, 'model_' + str(self.current_iteration) + '.pth'))
        self.network.to(self.settings['device'])

        
        save_dict    = {
                         'optimizer_state_dict':optimizer_state_dict,
                       }
        
        for it_thresh in ['enough_iter', 'max_iter']:
            if self.current_iteration == self.settings[it_thresh]:
                torch.save(save_dict, os.path.join(save_path, 'opt_' + it_thresh + str(self.current_iteration) + '.pth'))
            
        torch.save(save_dict, os.path.join(save_path, 'opt_' + str(self.current_iteration) + '.pth'))
        self.network.to(self.settings['device'])

    def save_summaries(self):
        save_path = os.path.join(self.settings['summaries_path'], self.exp_name, self.src_domain)

        np.savetxt(os.path.join(save_path, "source_loss"), self.source_loss_history, fmt="%.5f")
        np.savetxt(os.path.join(save_path, "adaptation_loss"), self.adaptation_loss_history, fmt="%.5f")
        np.savetxt(os.path.join(save_path, "target_accuracy"), self.target_acc_history, fmt="%.5f")
        np.savetxt(os.path.join(save_path, "distribution"), self.pseudo_target_dist, fmt="%.5f")

        plt.title('Source log loss for domain {}'.format(self.src_domain))
        plt.plot(np.log(self.source_loss_history))
        plt.savefig(os.path.join(save_path, "source_loss_plot"))
        plt.clf()
        
        plt.title('Adaptation log loss for domain {} -> {}'.format(self.src_domain, self.trgt_domain))
        plt.plot(np.log(self.adaptation_loss_history))
        plt.savefig(os.path.join(save_path, "adaptation_loss_plot"))
        plt.clf()

        plt.title('Target accuracy')
        plt.plot(self.it_history, self.target_acc_history)
        plt.savefig(os.path.join(save_path, "target_accuracy_plot"))
        plt.clf()

    def load_summaries(self):
        save_path = os.path.join(self.settings['summaries_path'], self.exp_name, self.src_domain)

        self.source_loss_history = np.loadtxt(os.path.join(save_path, "source_loss"))
        self.adaptation_loss_history = np.loadtxt(os.path.join(save_path, "adaptation_loss"))
        self.target_acc_history = np.loadtxt(os.path.join(save_path, "target_accuracy"))
        
    '''
    Utility Functions to initialize source and target dataloaders
    '''
    def get_all_train_src_dataloaders(self):
        self.initialize_src_train_dataloader()

    def initialize_src_train_dataloader(self):
        source_dataset_train = FeatureDataset("{}_{}.csv".format(self.src_domain, self.src_domain))
        self.source_dl_iter_train_list = iter(DataLoader(source_dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=5,drop_last=True,pin_memory=True))
        
    def get_all_val_target_dataloaders(self):
        self.initialize_target_val_dataloader()

    def initialize_target_val_dataloader(self):
        target_dataset_val = FeatureDataset("{}_{}.csv".format(self.src_domain, self.trgt_domain))
        self.target_dl_iter_val_list = iter(DataLoader(target_dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=5,pin_memory=True))

    def initialize_target_adapt_dataloader(self):
        adapt_target_dataset_train = FeatureDataset("{}_{}.csv".format(self.src_domain, self.trgt_domain))
        self.adapt_target_dl_iter_train_list = iter(DataLoader(adapt_target_dataset_train, batch_size=self.adapt_batch_size, shuffle=True, num_workers=5,drop_last=True,pin_memory=True))
        
    '''
    Utility function to set the model in eval or train mode
    '''
    def set_mode(self,mode):
        self.phase = mode

        if self.phase == self.settings['mode']['train']:
            for nc,compts in self.to_train.items():
                for name,val in compts.items():
                    if val:
                        self.network.model[nc][name].train()
                    else:
                        self.network.model[nc][name].eval()

        elif self.phase == self.settings['mode']['val']:
            self.network.eval()
        
    '''
    Initializing optimizers
    '''
    def init_optimizers(self):
        self.optimizer_dict  = {}
        to_train = self.settings['to_train']
        for loss_name,loss_details in self.settings['optimizer_dict'].items():
            if self.settings['use_loss'][loss_name]:
                opt_param_list = []
                for dom,cmpts in loss_details.items():
                    for comp in cmpts:
                        if to_train[dom][comp]:
                            if comp == 'G':
                                opt_param_list.append({'params':self.network.model[dom][comp].parameters(), 'lr':self.settings['lr'][loss_name] / 10.0, 'weight_decay':5e-4})           
                            else:
                                opt_param_list.append({'params':self.network.model[dom][comp].parameters(), 'lr':self.settings['lr'][loss_name], 'weight_decay':5e-4})          
                self.optimizer_dict[loss_name] = optim.Adam(params = opt_param_list)

    '''
    Target dataset validation
    '''
    def val_over_target_set(self, save_weights=True):
        self.set_mode(self.settings['mode']['val'])
        self.get_all_val_target_dataloaders()

        with torch.no_grad():
            dom = self.trgt_domain

            all_labels_trgt                  =  []
            all_preds_trgt                   =  []
            
            target_dl_iter_val_list = self.target_dl_iter_val_list
            
            for data in tqdm(target_dl_iter_val_list,desc=dom):
                indx,images,label,_         = data

                x                           = images.to(self.settings['device']).float()
                label                       = label.to(self.settings['device']).long()

                F                           = self.network.model['global']['Fs'](x)
                M                           = self.network.model['global']['M'](F)
                
                cls_logits,_,mat            = metrics.get_logits(feats={'M':M})
                cls_confs,cls_preds         = torch.max(cls_logits,dim=-1)

                all_labels_trgt.extend(list(label.cpu().numpy()))
                all_preds_trgt.extend(list(cls_preds.cpu().numpy()))
                
            if save_weights:
                self.check_and_save_weights(metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt}),dom)
                self.target_acc_history.append(metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt}))
                self.it_history.append(self.current_iteration)
            else:
                print("target accuracy at iteration {} = {}".format(self.current_iteration, metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt})))
            

            return metrics.get_metric('cls_acc',feats={'cls_labels':all_labels_trgt,'cls_preds':all_preds_trgt})
                
    '''
    Function to calculate the loss value
    '''
    def get_loss(self,which_loss):
        assert which_loss in ['source', 'target']

        if which_loss == 'source':


            src_M                   = self.src_features['M']
            src_labels              = self.src_features['label']

            loss                    = metrics.loss_CE(src_M/self.settings['softmax_temperature'], src_labels)
            
            if self.current_iteration % self.settings['log_interval'] == 0:
                print('iteration {}: {} loss = {} domain = {}'.format(self.current_iteration, which_loss, loss.cpu(), self.src_domain))

            self.source_loss_history.append(loss.item())
        elif which_loss == 'target':
            trgt_F                  = self.trgt_features['F']

            # Compute the number of gaussian samples to be used for the current batch
            normalized_dist = self.pseudo_target_dist / np.sum(self.pseudo_target_dist)
            num_samples = np.array(normalized_dist * self.adapt_batch_size, dtype=int)
            while self.adapt_batch_size > np.sum(num_samples):
                idx = np.random.choice(range(self.N_CLASSES), p = normalized_dist)
                num_samples[idx] += 1

            # Get gaussian samples for the current batch
            gz = []
            gy = []
            for c in range(self.N_CLASSES):
                ind = np.where(self.gaussian_y == c)[0]
                ind = ind[np.random.choice(range(len(ind)), num_samples[c], replace=False)]
                gz.append(self.gaussian_z[ind])
                gy.append(self.gaussian_y[ind])
            gz = np.vstack(gz)
            gy = np.concatenate(gy)

            gz = torch.as_tensor(gz).to(self.settings['device']).float()
            gy = torch.as_tensor(gy).to(self.settings['device']).long()

            loss = metrics.sliced_wasserstein_distance(trgt_F, gz, self.settings['num_projections'], 2, self.settings['device'])

            if self.current_iteration % self.settings['log_interval'] == 0:
                print('iteration {}: {} loss  ={} '.format(self.current_iteration, which_loss,loss.cpu()))

            self.adaptation_loss_history.append(loss.item())
        
        return loss
    
    '''
    Function to select active losses
    '''
    def loss(self):
        optim           = self.optimizer_dict[self.active_losses[self.current_loss]]
        optim.zero_grad()
        loss            = self.get_loss(self.active_losses[self.current_loss])
        loss.backward()
        optim.step()
                
    '''
    Function to implement the forward prop for a single source
    '''
    def forward(self):
        self.set_mode(self.settings['mode']['train'])

        if self.active_losses[self.current_loss] == 'source':
            # Computing the values for the source domain
            self.src_features                               = {}

            dom = self.src_domain
            images,label,domain_label                       = self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label']

            feats_F                                         = self.network.model['global']['Fs'](images)
            feats_M                                         = self.network.model['global']['M'](feats_F)

            self.src_features['F']              =  feats_F
            self.src_features['M']              =  feats_M
            self.src_features['label']          =  label
            self.src_features['domain_label']   =  domain_label
        elif self.active_losses[self.current_loss] == 'target':
            # Computing target domain info
            self.trgt_features                              = {}

            image_batch_concat                              = []
            label_batch_concat                              = []
            domain_label_batch_concat                       = []

            dom = self.trgt_domain
            images,label,domain_label                       = self.trgt_data[dom]['images'],self.trgt_data[dom]['label'],self.trgt_data[dom]['domain_label']

            # During adaptation, keep the feature extractor frozen
            feats_F                                         = self.network.model['global']['Fs'](images)
            feats_M                                         = self.network.model['global']['M'](feats_F)

            self.trgt_features['F']              =  feats_F
            self.trgt_features['M']              =  feats_M
            self.trgt_features['label']          =  label
            self.trgt_features['domain_label']   =  domain_label
    

    def get_target_pseudo_distribution(self, it_thresh='enough_iter'):
        logit_sum = None
        # it_thresh = 'enough_iter'

        for src_domain_idx in range(len(self.settings['src_datasets'])):
            # TODO: try this - don't combine distributions for the pseudodistribution construction
            if self.src_domain != self.settings['src_datasets'][src_domain_idx]:
                continue

            print("Loading trainer for source domain {}".format(self.settings['src_datasets'][src_domain_idx]))
            
            trainer_S = MultiSourceTrainer(src_domain_idx)
            # trainer_S.load_model_weights('model_' + it_thresh + str(trainer_S.settings[it_thresh]) + '.pth')
            trainer_S.load_model_weights(weights_file='model_' + it_thresh + str(trainer_S.settings[it_thresh]) + '.pth')
            trainer_S.set_mode(trainer_S.settings['mode']['val'])
            
            # [!IMPORTANT] Target val dataloader does not shuffle data points. Shuffling breaks the following code
            trainer_S.get_all_val_target_dataloaders()
            with torch.no_grad():
                # Gather samples from both source domains
                all_labels_tar                  =  []
                all_logits                      =  []

                dom = trainer_S.settings['trgt_datasets'][0]

                target_dl_iter_val_list = trainer_S.target_dl_iter_val_list
                for data in tqdm(target_dl_iter_val_list, desc=dom):
                    indx,images,label,_         = data
                    x                           = images.to(trainer_S.settings['device']).float()
                    label                       = label.to(trainer_S.settings['device']).long()
                    # G                           = trainer_S.network.model['global']['G'](x)
                    F                           = trainer_S.network.model['global']['Fs'](x)
                    M                           = trainer_S.network.model['global']['M'](F)

                    cls_logits,_,mat            = metrics.get_logits(feats={'M':M})
                    all_logits.append(cls_logits)
                    
                    all_labels_tar.extend(list(label.cpu().numpy()))
                    
                all_labels_tar = np.asarray(all_labels_tar)
                all_logits = torch.cat(all_logits, dim=0).cpu().numpy()

                if logit_sum is None:
                    logit_sum = all_logits
                else:
                    logit_sum += all_logits

        print("Source-only accuracy is {}".format(np.mean(all_labels_tar == np.argmax(logit_sum, axis=-1))))

        print('y__hat counts', np.unique(np.argmax(logit_sum, axis=-1), return_counts=True)[1])
        print('y_true counts', np.unique(all_labels_tar, return_counts=True)[1])

        conf = np.asarray([x / np.sum(x) for x in logit_sum])
        dist = np.sum(conf, axis=0)
        
        return dist, conf, all_labels_tar


    '''
    Function for training the the data
    This function is called at every iteration
    '''
    def train (self):
        self.src_data                           = {}
        self.trgt_data                          = {}

        self.current_loss = 0
        if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
            self.current_loss = 1

        cond_1 = self.active_losses[self.current_loss] not in self.settings['losses_after_enough_iters']
        cond_2 = self.current_iteration <= max(self.settings['val_after'],self.settings['enough_iter'])
        
        if (cond_1 and cond_2) or (not cond_2):
            if self.current_iteration <= max(self.settings['val_after'],self.settings['enough_iter']):
                # Extract a batch of source samples
                dom = self.src_domain
                try:
                    self.src_data[dom]={}
                    _,self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label'] = self.source_dl_iter_train_list.next()
                    self.src_data[dom]['images'] = Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
                    self.src_data[dom]['label'] = Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
                    self.src_data[dom]['domain_label'] = Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()
                except StopIteration:
                    self.initialize_src_train_dataloader()
                    self.src_data[dom]={}
                    _,self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label'] = self.source_dl_iter_train_list.next()
                    self.src_data[dom]['images'] = Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
                    self.src_data[dom]['label'] = Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
                    self.src_data[dom]['domain_label'] = Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()
            else:                
                # Distribution matching between target domain and gaussians
                if self.current_iteration == max(self.settings['val_after'], self.settings['enough_iter']) + 1:
                    print("STARTING ADAPTION FOR MODEL TRAINED ON {}".format(self.src_domain))

                    # If there are high confidence pseudo-labels for most of the samples, use those for adaptation
                    self.pseudo_target_dist, conf, _ = self.get_target_pseudo_distribution()
                    if np.sum(np.max(conf, axis=-1) > self.settings['confidence_thresh']) / conf.shape[0] < self.settings['confidence_ratio']:
                        self.pseudo_target_dist = np.ones(self.N_CLASSES, dtype=int)

                    print("FINALIZED DISTRIBUTION FOR ADAPTATION ON {}:".format(self.trgt_domain))
                    print(self.pseudo_target_dist)
                    print()

                    # Learn gaussians
                    gaussian_utils.learn_gaussians(self)

                    n_samples = np.ones(self.N_CLASSES, dtype=int) * self.settings["gaussian_samples_per_class"]
                    self.gaussian_z, self.gaussian_y = gaussian_utils.sample_from_gaussians(self.means, self.covs, n_samples)

                    self.initialize_target_adapt_dataloader()

                # Resample gaussians to counter over-fitting
                if self.current_iteration % (self.settings['max_iter'] // 10) == 0:
                    n_samples = np.ones(self.N_CLASSES, dtype=int) * self.settings["gaussian_samples_per_class"]
                    self.gaussian_z, self.gaussian_y = gaussian_utils.sample_from_gaussians(self.means, self.covs, n_samples)

                dom = self.trgt_domain
                try:
                    self.trgt_data[dom]={}
                    _,self.trgt_data[dom]['images'],self.trgt_data[dom]['label'],self.trgt_data[dom]['domain_label'] = self.adapt_target_dl_iter_train_list.next()
                    self.trgt_data[dom]['images'] = Variable(self.trgt_data[dom]['images']).to(self.settings['device']).float()
                    self.trgt_data[dom]['label'] = Variable(self.trgt_data[dom]['label']).to(self.settings['device']).long()
                    self.trgt_data[dom]['domain_label'] = Variable(self.trgt_data[dom]['domain_label']).to(self.settings['device']).long()
                except StopIteration:
                    self.initialize_target_adapt_dataloader()
                    self.trgt_data[dom]={}
                    _,self.trgt_data[dom]['images'],self.trgt_data[dom]['label'],self.trgt_data[dom]['domain_label'] = self.adapt_target_dl_iter_train_list.next()
                    self.trgt_data[dom]['images'] = Variable(self.trgt_data[dom]['images']).to(self.settings['device']).float()
                    self.trgt_data[dom]['label'] = Variable(self.trgt_data[dom]['label']).to(self.settings['device']).long()
                    self.trgt_data[dom]['domain_label'] = Variable(self.trgt_data[dom]['domain_label']).to(self.settings['device']).long()

            self.forward()
            self.loss()
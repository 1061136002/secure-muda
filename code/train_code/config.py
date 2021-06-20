import os
import shutil
import torch
import glob
import pdb
import numpy as np
from config_populate import data_settings

# While config is mostly unchanged throughout different runs, to avoid creating multiple config files for
# each problem, we select experiments using this auxiliary file
import exp_select as exp_select

def gen_exp_name():
    st ='expt'
    st = '_'.join([st,str(settings['bb'])])
    st = '_'.join([st,dataset_name])
    st = '_'.join([st,data_key])
    st = '_'.join([st, str(exp_select.exp_id)])

    if len(settings['optimizer_dict'] )>0:
        active_losses = [loss for loss in settings['optimizer_dict'] if settings['use_loss'][loss] == True ]

    if len(settings['id_str'])>0:
        st = '_'.join([st,settings['id_str']])

    return st



settings                                = {}

server_root_path                        = '../../'
dataset_name                            = exp_select.dataset_name
data_key                                = exp_select.data_key

settings['server_root_path']            = server_root_path
settings['dataset_dir']                 = os.path.join('data', 'pretrained-features', dataset_name)


settings['quiet_mode']                  = True

#dataset settings
settings['C']                           = data_settings[dataset_name][data_key]['C']
settings['C_dash']                      = data_settings[dataset_name][data_key]['C_dash']
settings['num_C_dash']                  = data_settings[dataset_name][data_key]['num_C_dash']

settings['num_C']                       = data_settings[dataset_name][data_key]['num_C']
settings['src_datasets']                = data_settings[dataset_name][data_key]['src_datasets']
settings['trgt_datasets']               = data_settings[dataset_name][data_key]['trgt_datasets']

st0 = np.random.get_state()[1][0]
t0  = torch.initial_seed()
settings['seed_value']                  = {'torch':t0,'np':st0}

settings['resolution']                  = 224


settings['index_list']                  = 'index_list'
settings['balance_dataset']             = False


settings['bb']                          = 'resnet50'
settings['bb_output']                   = 2048  
settings['F_dims']                      = 256


settings['to_train']                    = {

                                            'global': {
                                                    'G' : True,
                                                    'Fs': True,
                                                    'M' : True,
                                                    }
                                            
                                          }


settings['softmax_temperature']         = 1


settings['use_loss']                    = { 
                                            'source':True,
                                            'target':True,
                                          }

settings['losses_after_enough_iters']   = ['target']

#optimizer settings
settings['optimizer_dict']              = { 
                                            'source':{'global': ['G', 'Fs', 'M']},
                                            'target':{'global': ['G', 'Fs', 'M']}
                                          }


settings['lr']                          = {
                                            'source':1e-5 if dataset_name != 'office-home' else 1e-6,
                                            'target':1e-5 if dataset_name != 'office-home' else 1e-6,
                                          }


settings['tau']                         = 0.0     # Threshold for gaussian samples
settings['gaussian_samples_per_class']  = 1000
settings['num_projections']             = 100
settings['confidence_thresh']           = .8
settings['confidence_ratio']            = .8
settings['eval_tuning_steps']           = 2000

#training settings
settings['id_str']                      = ''
settings['recalc_pseudo_labels']        = False
settings['topK']                        = 0
settings['exp_name']                    = gen_exp_name()

settings['validate_target_before_enough_iters'] = True

settings['mode']                        = {'train':0,'val':1}
settings['summaries_path']              = os.path.join(server_root_path, 'summaries')
settings['weights_path']                = os.path.join(server_root_path, 'weights')
settings['gaussians_path']              = os.path.join(server_root_path, 'computed_gaussians')
  
settings['gpu']                         =  exp_select.gpu
settings['device']                      = 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])
settings['tb_port_no']                  = 9999 - settings['gpu']

settings['checkpoints_count']           = 1

settings['expt_dict']                   = {
                                            'office-31':{
                                                            'AD_W':{'enough_iter':5000,'max_iter':12000,'val_after':500,'batch_size':16,'adapt_batch_size':62,'val_batch_size_factor':20},
                                                            'DW_A':{'enough_iter':5000,'max_iter':12000,'val_after':500,'batch_size':16,'adapt_batch_size':62,'val_batch_size_factor':20},
                                                            'AW_D':{'enough_iter':5000,'max_iter':12000,'val_after':500,'batch_size':16,'adapt_batch_size':62,'val_batch_size_factor':20}
                                                        },
                                            'image-clef':{
                                                            'PC_I':{'enough_iter':1000,'max_iter':2000,'val_after':100,'batch_size':16,'adapt_batch_size':24,'val_batch_size_factor':20},
                                                            'IC_P':{'enough_iter':1000,'max_iter':2000,'val_after':100,'batch_size':16,'adapt_batch_size':24,'val_batch_size_factor':20},
                                                            'IP_C':{'enough_iter':1000,'max_iter':2000,'val_after':100,'batch_size':16,'adapt_batch_size':24,'val_batch_size_factor':20}
                                                        },
                                            'office-home':{
                                                            'ACP_R':{'enough_iter':20000,'max_iter':50000,'val_after':1000,'batch_size':160,'adapt_batch_size':195,'val_batch_size_factor':20},
                                                            'ACR_P':{'enough_iter':20000,'max_iter':50000,'val_after':1000,'batch_size':160,'adapt_batch_size':195,'val_batch_size_factor':20},
                                                            'APR_C':{'enough_iter':20000,'max_iter':50000,'val_after':1000,'batch_size':160,'adapt_batch_size':195,'val_batch_size_factor':20},
                                                            'CPR_A':{'enough_iter':20000,'max_iter':50000,'val_after':1000,'batch_size':160,'adapt_batch_size':195,'val_batch_size_factor':20}
                                                        },
                                            'office-caltech':{
                                                            'ACD_W':{'enough_iter':2000,'max_iter':3500,'val_after':100,'batch_size':16,'adapt_batch_size':50,'val_batch_size_factor':10},
                                                            'ADW_C':{'enough_iter':2000,'max_iter':3500,'val_after':100,'batch_size':16,'adapt_batch_size':50,'val_batch_size_factor':10},
                                                            'ACW_D':{'enough_iter':2000,'max_iter':3500,'val_after':100,'batch_size':16,'adapt_batch_size':50,'val_batch_size_factor':10},
                                                            'CDW_A':{'enough_iter':2000,'max_iter':3500,'val_after':100,'batch_size':16,'adapt_batch_size':50,'val_batch_size_factor':10}
                                                        }
                                          }

settings['log_interval']                = settings['expt_dict'][dataset_name][data_key]['val_after']
settings['start_iter']                  = 0
settings['max_iter']                    = settings['expt_dict'][dataset_name][data_key]['max_iter']
settings['enough_iter']                 = settings['expt_dict'][dataset_name][data_key]['enough_iter']
settings['val_after']                   = settings['expt_dict'][dataset_name][data_key]['val_after']
settings['batch_size']                  = settings['expt_dict'][dataset_name][data_key]['batch_size']
settings['adapt_batch_size']            = settings['expt_dict'][dataset_name][data_key]['adapt_batch_size']
settings['val_batch_size_factor']       = settings['expt_dict'][dataset_name][data_key]['val_batch_size_factor']
settings['load_model']                  = False
settings['load_opt']                    = False


settings['continue_training']           = False
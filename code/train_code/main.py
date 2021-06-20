import numpy as np
import os

import config
from trainer import MultiSourceTrainer

from evaluate_model import learn_w_w2, learn_w_generalizability, get_target_accuracy, get_individual_performance

def run_trainer():
    # First, perform source-only training, in order to have source-only models for each domain.
    trainers = {}

    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]

        print("Training on source domain {}".format(config.settings['src_datasets'][src_domain_idx]))
        
        trainers[dom] = MultiSourceTrainer(src_domain_idx)
        trainers[dom].init_folder_paths()

        while trainers[dom].current_iteration < config.settings['enough_iter'] + 1:
            trainers[dom].train()

            if trainers[dom].current_iteration%trainers[dom].settings['val_after']== 0:
                trainers[dom].set_mode(config.settings['mode']['val'])
                target_acc = trainers[dom].val_over_target_set()
                print("Obtained target accuracy = {} at iteration {}".format(target_acc, trainers[dom].current_iteration))
                
            trainers[dom].current_iteration += 1


    # Now, perform adaptation
    for src_domain_idx in range(len(config.settings['src_datasets'])):
        dom = config.settings['src_datasets'][src_domain_idx]
        print("Training on source domain {}".format(dom))

        while trainers[dom].current_iteration < config.settings['max_iter'] + 1:
            trainers[dom].train()

            if trainers[dom].current_iteration%trainers[dom].settings['val_after']== 0:
                trainers[dom].set_mode(config.settings['mode']['val'])
                target_acc = trainers[dom].val_over_target_set()
                print("Obtained target accuracy = {} at iteration {}".format(target_acc, trainers[dom].current_iteration))

            if trainers[dom].current_iteration == trainers[dom].settings['max_iter']:
                trainers[dom].load_model_weights(it_thresh='max_iter')
                trainers[dom].load_optimizers()
                target_acc_loaded = trainers[dom].val_over_target_set(save_weights=False)
                
                print("Final target accuracy for domain {} = {}".format(trainers[dom].src_domain, target_acc.round(3)))
                assert np.abs(target_acc - target_acc_loaded) < 1e-8

                trainers[dom].save_summaries()
                
            trainers[dom].current_iteration += 1

def evaluate():
    combined_target_perf = []
    
    eval_tuning_steps = config.settings['eval_tuning_steps']

    # Combine domains based on a generalizability objective
    w_hat_gen = learn_w_generalizability(num_steps=eval_tuning_steps)
    combined_target_perf.append(get_target_accuracy(w_hat_gen, 'max_iter')[0])

    # Combine domains based on W2 distance between latent spaces
    w_hat_w2 = learn_w_w2(num_steps=eval_tuning_steps)
    combined_target_perf.append(get_target_accuracy(w_hat_w2, 'max_iter')[0])

    # Combine domains evenly
    combined_target_perf.append(get_target_accuracy(np.ones(len(w_hat_w2)), 'max_iter')[0])

    individual_src_only_perf, individual_target_perf = get_individual_performance()

    return combined_target_perf, individual_src_only_perf, individual_target_perf

def main():
    run_trainer()

    print("\n\n\n")

    combined, individual_src_only, individual_target = evaluate()

    print("\n\n\n")

    all_results = np.concatenate([combined, individual_src_only, individual_target])
    all_results = all_results.reshape(1, -1)
    np.savetxt(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'results.txt'), all_results, fmt='%.5f', newline = ' ')


if __name__ == "__main__":
    main()
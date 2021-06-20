#!/usr/bin/env python

import argparse
import os

parser = argparse.ArgumentParser(description='Run training code for a user chosen dataset in parallel')
parser.add_argument('--dataset', type=str, default="office-31",
                    help='dataset to run the code on')
parser.add_argument('--task', type=str, default="DW_A",
                    help='the adaptation task to perform')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='the id of the gpu to be used')
parser.add_argument('--num_exp', type=int, default=5,
                    help='number of runs for each dataset adaptation task')
args = vars(parser.parse_args())

def populate_exp_select(dataset, task, gpu_id, exp_id):
	with open("exp_select.py", 'w') as f:
		f.write("dataset_name='{}'\n".format(dataset))
		f.write("data_key='{}'\n".format(task))
		f.write("gpu={}\n".format(gpu_id))
		f.write("exp_id={}\n".format(exp_id))


def main():
	print(args)

	for exp_id in range(args['num_exp']):
		populate_exp_select(args['dataset'], args['task'], args['gpu_id'], exp_id)
		os.system("python main.py".format(args['dataset'], args['task']))

if __name__ == "__main__":
	main()
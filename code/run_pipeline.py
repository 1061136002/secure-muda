# Main experiment runner. Example call: python run_pipeline.py --dataset image-clef --num_exp 10 --gpu_id 0

import argparse
import os


import subprocess

from multiprocessing import Process

from train_code.config_populate import data_settings

parser = argparse.ArgumentParser(description='Run training code for a user chosen dataset in parallel')
parser.add_argument('--dataset', type=str, default="office-31",
                    help='dataset to run the code on')
parser.add_argument('--num_exp', type=int, default=5,
                    help='number of runs for each dataset adaptation task')
parser.add_argument('--gpu_id', type=int, default=0, 
					help='id of the gpu to perform experiments on. If set to -1, each different task will be performed on an individual gpu.')

args = vars(parser.parse_args())

def start_runner(dataset, task, gpu_id, num_exp):
	subprocess.Popen(["./runner.py", '--dataset', dataset, '--task', task, '--gpu_id', str(gpu_id), '--num_exp', str(num_exp)], \
		cwd="./train_code_{}_{}".format(dataset, task))

def main():
	print(args)

	dataset = args['dataset']
	num_exp = args['num_exp']

	print(data_settings[dataset].keys())

	# For each task available for the current dataset, copy the code folder
	for task in data_settings[dataset].keys():
		os.system("rm -rf train_code_{}_{}".format(dataset, task))
		os.system("cp -r train_code train_code_{}_{}".format(dataset, task))

	# Run num_exp experiments for each task. Do it in parallel.
	for task in data_settings[dataset].keys():
		if args['gpu_id'] == -1:
			gpu_id = list(data_settings[dataset].keys()).index(task)
		else:
			gpu_id = args['gpu_id']

		start_runner(dataset, task, gpu_id, num_exp)


if __name__ == "__main__":
	main()

import os
from multiprocessing import Process


dataset   = 'office-home'
domains   = ['Art', 'Clipart', 'Product', 'Real\ World']
n_classes = 65
n_epochs  = 200

def run_cmd(src, trgt, gpu_id):
	cmd = "python main.py --dataset {} --dataset_path '../../data/{}/' --model_name resnet50 --source {} --target {} --num_class {} --epoch {} --gpu {}".format(\
		dataset, dataset, src, trgt, n_classes, n_epochs, gpu_id)
	print(cmd)
	os.system(cmd)

	if not os.path.exists("../../data/pretrained-features/{}".format(dataset)):
		os.mkdir("../../data/pretrained-features/{}".format(dataset))
	os.system("cp ./save_model_{}/{}_{}_resnet50.csv ../../data/pretrained-features/{}/{}_{}.csv".format(dataset, src, trgt, dataset, src, trgt))

def get_target_feature_for_fixed_src(src):
	for d2 in domains:
		gpu_id = domains.index(src)
		run_cmd(src, d2, gpu_id)


proc = []
for d1 in domains:
	proc.append(Process(target=get_target_feature_for_fixed_src, args=([d1])))
	proc[-1].start()

for p in proc:
	p.join()
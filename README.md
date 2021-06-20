# Code for Secure Domain Adaptation with Multiple Sources

We attach the code for Secure Domain Adaptation with Multiple Sources. The code has been tested with **PyTorch 1.7.1+cu110** and **Python 3.8**.

## Dataset preparation

The repository comes with pre-trained features already made available in the data folder. If you prefer to generate these from scratch, please refer to the README in the data folder.

### Additional datasets

If you wish to evaluate the code on a new dataset, you would need to add it to the data folder following the schematic described there, and changing the **config.py** and **config_populate.py** files in the **code/train_code** folder accordingly.

## Configuring the run

Before running the model, runtime parameters can be configured via the **config.py** file.

## Running the code

The code assumes the presence of a GPU. To run the code on a specific task pertaining to one of the four supported datasets, go to **code/train_code** and call the **runner.py** script.
For example, to run the **DW_A** task for **office-31**, call

```
	python runner.py --dataset office-31 --task DW_A --gpu_id 0 --num_exp 1
```

The above command will run the specified task once on GPU 0. To re-run the task multiple times, simply specify a larger number of experiments via the **num_exp** flag.

After finishing a run, results are stored in the summaries and weights folders in the repository root. Each experiment will have a **results.txt** file, whose output format is
described in the **evaluate** function in **main.py** (the score based on SWD aggregation is the second one in the file).

### Multi-gpu training

It is possible to run several tasks in parallel if more than 1 GPUs are present on the system. You will need to use the **run_pipeline.py** script in the **code** folder. For example,
running all the tasks for **office-31** exactly once on 3 GPUs (1 task on each GPU), simply run

```
	python run_pipeline.py --dataset office-31 --num_exp 1 --gpu_id -1

```

## Acknowledgements

We thank the authors of ["Your Classifier can Secretly Suffice Multi-Source Domain Adaptation"](https://sites.google.com/view/simpal) for making their codebase available, as we use it as
backbone for our codebase. We use the implementation if Sliced Wasserstein Distance from ["Sliced Wasserstein Autoencoder"](https://github.com/eifuentes/swae-pytorch). Finally, the code
for generating pre-trained features is adapted from ["Transfer Learning"](https://github.com/jindongwang/transferlearning), a repository containing many useful reseach tools for this area.

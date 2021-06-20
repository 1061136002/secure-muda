import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import config as config
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# https://medium.com/@shashikachamod4u/excel-csv-to-pytorch-dataset-def496b6bcc1
class FeatureDataset(Dataset):

    def __init__(self, file_name):

        file_name = os.path.join(config.server_root_path, config.settings['dataset_dir'], file_name)

        data = pd.read_csv(file_name)
        data = np.asarray(data)

        x = data[:, :-1]
        y = data[:, -1]

        self.img = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return idx, self.img[idx], self.label[idx], self.label[idx]
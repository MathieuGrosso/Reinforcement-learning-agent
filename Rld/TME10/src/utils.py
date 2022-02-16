import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from torch import optim
import numpy as np 
import datetime

class MNIST(Dataset):
    def __init__(self, images, labels, device, transform=None):
        # copy the np arrays to remove the UserWarning, as they are not writeable
        self.images = torch.tensor(images.copy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.copy(), dtype=torch.int64)
        self.images /= 255.0
        self.transform = transform

        _, self.width, self.height = self.images.shape
        self.n_features = self.width * self.height

        # Use GPU if available
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
    

    def __getitem__(self, index):
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.images)



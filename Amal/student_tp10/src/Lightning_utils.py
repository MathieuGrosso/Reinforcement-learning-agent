import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamaestro import prepare_dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.distributions import Categorical
from torch.functional import norm
from torch.utils.data import DataLoader, TensorDataset, random_split

BATCH_SIZE = 300
TRAIN_RATIO = 0.8
LOG_PATH = "./logs"


class LightningNetwork(pl.LightningModule):
    def __init__(self,
                 name,
                 network: nn.Module,
                 learning_rate = 1e-3,
                 log_freq : Optional[int] = None,
                 warmup = 100,
                 max_iters = 1500):
        super().__init__()
        self.model = network
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = name
        self.log_freq = log_freq
        self.warmup = warmup
        self.max_iters = max_iters

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
    
        return optimizer

    def training_step(self,batch,batch_idx):
        x, y = batch
        if self.log_freq is not None:
            yhat, attention_weights = self(x)
        else:
            yhat = self(x)
        loss = self.loss(yhat,y)

        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        self.log("training_loss",loss, on_step=False,on_epoch=True)

        # Log entropy
        if self.log_freq is not None and self.global_step % self.log_freq == 0:
            random_logits = torch.randn_like(attention_weights)
            self.logger.experiment.add_histogram('entropy_output', Categorical(logits=attention_weights).entropy(),
                                                 self.global_step)
            self.logger.experiment.add_histogram('entropy_random_output', Categorical(logits=random_logits).entropy(),
                                                 self.global_step)
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        if self.log_freq is not None:
            yhat, _ = self(x)
        else:
            yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x), on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation_loss",loss, on_step=False,on_epoch=True)
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def test_epoch_end(self, outputs):
        pass
        
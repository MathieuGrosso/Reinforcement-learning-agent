import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from icecream import ic
import optuna
from optuna.integration import PyTorchLightningPruningCallback

BATCH_SIZE = 311
TRAIN_RATIO = 0.8
LOG_PATH = "/tmp/runs/lightning_logs"


class Net(nn.Module):
    def __init__(self,input_size,hidden_dim,output_size,dropout):
        super(Net,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,output_size)
        self.activation = nn.ReLU()
        self.dropout_factor = dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self,x):
        "model changed to add dropout, regularisation and batchnorm"
        x1 = self.linear1(x)
        x1 = self.dropout1(x1)
        x2 = self.linear2(x1)
        x2 = self.dropout2(x2)
        out   = self.activation(self.linear3(x2))
            
        return out 
    
    
    



class Lit2Layer(pl.LightningModule):
    def __init__(self,dim_in,l,dim_out,learning_rate=1e-3,dropout=0.5,reg=None):
        super().__init__()
        self.model = Net(dim_in,100,dim_out,dropout = dropout)
        self.reg = reg
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "exemple-lightning"

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = self.loss(yhat,y)
        if self.reg is not None: 
            lambdareg = self.reg
            l1_penalty = sum(p.abs().sum() for p in self.parameters())
            loss = loss + lambdareg * l1_penalty
        # loss = self.loss(yhat,y)
        # if self.reg is not None: 
        #     l1_penalty = 0
        #     for param in self.parameters():
        #         l1_penalty += torch.sum(torch.abs(param))
        #     loss = nn.functional.cross_entropy(yhat, y) + self.l1_reg * l1_penalty
        # else:
        #     loss = nn.functional.cross_entropy(yhat, y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x),on_step=False,on_epoch=True)
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def training_epoch_end(self,outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/train":total_loss,f"acc/train":total_acc})
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX
    def validation_epoch_end(self, outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/train":total_loss,f"acc/train":total_acc})

    def test_epoch_end(self, outputs):
        pass




class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        prepare_dataset("com.lecun.mnist")

    def setup(self,stage=None):
        ds = prepare_dataset("com.lecun.mnist")
        if stage =="fit" or stage is None:
            # Si on est en phase d'apprentissage
            shape = ds.train.images.data().shape
            self.dim_in = shape[1]*shape[2]
            self.dim_out = len(set(ds.train.labels.data()))
            ds_train = TensorDataset(torch.tensor(ds.train.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.train.labels.data()).long())
            train_length = int(shape[0]*self.train_ratio)
            self.mnist_train, self.mnist_val, = random_split(ds_train,[train_length,shape[0]-train_length])
        if stage == "test" or stage is None:
            # en phase de test
            self.mnist_test= TensorDataset(torch.tensor(ds.test.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.test.labels.data()).long())

    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)


# data = LitMnistData()

# data.prepare_data()
# data.setup(stage="fit")

# model = Lit2Layer(data.dim_in,10,data.dim_out,learning_rate=1e-3)

# logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)

# trainer = pl.Trainer(default_root_dir=LOG_PATH,logger=logger,max_epochs=100)
# trainer.fit(model,data)
# trainer.test(model,data)

"""what are callback in pytorch lightning: during the flow --> you can execute them. It is a general idea of computer vision where during hte flow of the execution, Lightning has a callback system to execute them when needed. Callbacks should capture NON-ESSENTIAL logic that is NOT required for your lightning module to run. can do what you want/imagine, design any callback...
 a hook : is a function that can be call at some point during training. 
 """


def objective(trial):
    #parameters to optimize : 
    dropout_trial = trial.suggest_float("dropout_trial",0.2,0.5)
    reg_trial = trial.suggest_float('reg_trial', low=0.0001, high=0.1)
    data = LitMnistData()
    data.prepare_data()
    data.setup(stage="fit")
    model = Lit2Layer(data.dim_in,10,data.dim_out,learning_rate=1e-3,dropout=dropout_trial,reg = reg_trial)

    logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)
    trainer = pl.Trainer(default_root_dir=LOG_PATH,logger = logger, max_epochs=100,gpus=1 if torch.cuda.is_available() else None,
    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy")])

    hyperparameters = dict(dropout=dropout_trial,reg=reg_trial)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return trainer.callback_metrics["val_accuracy"].item()

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100,timeout=60000)
    print("number of finished trials:{}".format(len(study.trials)))
    print('best trial:')
    trial=study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
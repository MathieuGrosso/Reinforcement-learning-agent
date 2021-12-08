# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        q=len(y)
        ctx.save_for_backward(yhat, y)
        #  TODO:  Renvoyer la valeur de la fonction
        return (1/q*torch.sum((y-yhat)**2))



    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        q=len(y)
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        #grad output : gradient de L 
        return grad_output * (-2/q*(y-yhat)), grad_output*(2/q*(y-yhat))


#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    @staticmethod
    def forward(ctx,X,W,b):
        ctx.save_for_backward(X,W,b)
        return X @ W + b

    @staticmethod
    def backward(ctx,grad_output):
        X,W,b=ctx.saved_tensors
        dX = grad_output@W.T
        dW = X.T@grad_output
        db = grad_output.sum(0)

        return dX, dW, db



## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply


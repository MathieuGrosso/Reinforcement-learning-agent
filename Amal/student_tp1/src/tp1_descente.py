import torch
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context, mse, linear
from icecream import ic

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    context_linear=Context()
    context_MSE=Context()
    yhat=Linear.forward(context_linear,x,w,b)
    loss=MSE.forward(yhat,y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    grad_output = torch.tensor(1, dtype=torch.float64)
    grad_output_yhat,grad_output_y = MSE.backward(ctx_MSE,grad_output)
    grad_x,grad_w,grad_b = Linear.backward(ctx_Linear,grad_output_yhat)
    ##  TODO:  Mise à jour des paramètres du modèle
    w=w-self.epsilon*grad_w
    b=b-self.epsilon*grad_b


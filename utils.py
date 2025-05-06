import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def getDevice():
    '''Function to get the device to be used for training the model. The priority is given to the GPU'''
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


class EarlyStopping:
    """Early stopping feature to stop the training when the validation loss stops
        decrasing. The priority is given to the learning rate scheduler; if the 
        learning rate is changed by the scheduler in the last `patience` number of
        epochs, the EarlyStopping doesn't stop the training."""

    def __init__(self, patience=5, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.early_stopping = False

    def __call__(self, val_loss, lr):
        if len(lr) < self.patience:
            None
        elif lr[-self.patience] != lr[-1]:
            None
        else:
            if abs(val_loss[-2] - val_loss[-1]) < self.delta:
                self.early_stopping = True
        return self.early_stopping


class WeightedLoss(nn.Module):
    """
    WeightedLoss is a custom loss function that takes a given loss function, disables
    first the reduction operation which is mean opearation by default. The it multiplies the
    output of the loss function by the weights and then returns the mean of the weighted loss.
    """

    def __init__(self, loss_function):
        super().__init__()
        self.loss_func = loss_function(reduction='none')

    def forward(self, y, t, w):
        loss = self.loss_func(y, t) * w
        return loss.mean()


class WeightedMSELoss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction: str = "none") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction) * weight

import torch
from torch import nn
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def init_loss_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch_losses = []
    epoch_training_losses = []
    epoch_validation10_losses = []
    epoch_validation90_losses = []
    mse_loss = torch.nn.MSELoss()
    
    return optimizer, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, mse_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR



def get_adam(
    model: nn.Module, 
    learning_rate: float = 0.001, 
    weight_decay: float = 0.0005
) -> optim.Optimizer:
    """
    Returns an Adam optimizer for the given model with specified learning rate and weight decay.

    Args:
        model (nn.Module): The model to optimize.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.

    Returns:
        optim.Optimizer: An Adam optimizer configured for the model.
    """
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_step_lr_scheduler(
    optimizer: optim.Optimizer, 
    step_size: int = 10, 
    gamma: float = 0.1
) -> StepLR:
    """
    Returns a StepLR scheduler for the given optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer to schedule.
        step_size (int): Number of epochs after which to decrease the learning rate.
        gamma (float): Factor by which to decrease the learning rate.

    Returns:
        StepLR: A StepLR scheduler configured for the optimizer.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
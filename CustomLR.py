import torch
from torch import nn
class CustomLRAdamOptimizer:
    """
        Implementation of the in the report mentioned learning rate that first rises for the amount of set warmup_steps and then declines and converges

    """

    def __init__(self, optimizer, model_dimension, num_of_warmup_steps, factor, current_step_number = 0):
        self.factor = factor
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps
        self.current_step_number = current_step_number

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for param in self.optimizer.param_groups:
            param['lr'] = current_learning_rate

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_current_learning_rate(self):
        # For readability purpose
        step = self.current_step_number
        warmup = self.num_of_warmup_steps

        return (self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))) * self.factor


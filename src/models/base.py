import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def generate(self):
        raise NotImplementedError('You have to implement generate method for your model')

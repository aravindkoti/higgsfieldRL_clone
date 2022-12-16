import wandb
from math import exp

class Clip():
    def __call__(self, module):
        if hasattr(module, 'gamma'):
            module.gamma.data.clamp_(0.0, 1.0)


def Sigmoid(x):
    y = 1 / (1+ exp(-5*(x-0.5)))
    return y



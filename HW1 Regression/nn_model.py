from torch import dropout
from myLib.Importer import *
from myLib.Utility import *
from config import *

class My_Model(nn.Module):

    def __init__(self, input_dim):
        super(My_Model, self).__init__()

        self.layers = nn.Sequential (
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 1),
        )


    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x
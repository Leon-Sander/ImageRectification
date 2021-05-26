import torch
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl
from models.unetnc import Estimator3d
from models.densenetccnl import dnetccnl 

class crease(pl.LightningModule):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False): #img_size
        super(crease, self).__init__()

        self.estimator3d = Estimator3d(input_nc = 3, output_nc = 8, num_downs = 5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.backward_map_estimator = dnetccnl(img_size=128, in_channels=3, out_channels=2, filters=32,fc_units=100)
        


    def forward(self, input):
        x = self.estimator3d.forward(input)
        output = self.backward_map_estimator(x)
        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch

        
import torch
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl
from models.unetnc import Estimator3d
from models.densenetccnl import dnetccnl 

class crease(pl.LightningModule):
    def __init__(self, input_nc, output_nc, num_downs, img_size = 256 , use_pre_trained = False, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False): #img_size
        super(crease, self).__init__()

        if use_pre_trained:
            self.estimator3d = torch.load('models/pretrained/estimator3d')
            self.backward_map_estimator = torch.load('models/pretrained/backward_map_estimator')
        else:
            self.estimator3d = Estimator3d(input_nc = 3, output_nc = 8, num_downs = 5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
            self.backward_map_estimator = dnetccnl(img_size, in_channels=3, out_channels=2, filters=32,fc_units=100)
        


    def forward(self, input):
        x = self.estimator3d.forward(input)
        output = self.backward_map_estimator(x)
        return output

    def loss_calculation(self, images, labels):
        return
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
# Densenet decoder encoder with intermediate fully connected layers and dropout

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np
import pytorch_lightning as pl
import angles
import math
import utils
from icecream import ic
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sys


def add_coordConv_channels(t):
    n,c,h,w=t.size()
    xx_channel=np.ones((h, w))
    xx_range=np.array(range(h))
    xx_range=np.expand_dims(xx_range,-1)
    xx_coord=xx_channel*xx_range
    yy_coord=xx_coord.transpose()

    xx_coord=xx_coord/(h-1)
    yy_coord=yy_coord/(h-1)
    xx_coord=xx_coord*2 - 1
    yy_coord=yy_coord*2 - 1
    xx_coord=torch.from_numpy(xx_coord).float()
    yy_coord=torch.from_numpy(yy_coord).float()

    if t.is_cuda:
    	xx_coord=xx_coord.cuda()
    	yy_coord=yy_coord.cuda()

    xx_coord=xx_coord.unsqueeze(0).unsqueeze(0).repeat(n,1,1,1)
    yy_coord=yy_coord.unsqueeze(0).unsqueeze(0).repeat(n,1,1,1)

    t_cc=torch.cat((t,xx_coord,yy_coord),dim=1)

    return t_cc



class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

# Dense block in encoder.
class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        return self.main(inputs)


class waspDenseEncoder256(nn.Module):
    def __init__(self, nc=1, ndf=32, ndim=256, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Tanh,
                 f_args=[]):
        super(waspDenseEncoder256, self).__init__()
        self.ndim = ndim

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

            # state size. (ndf) x 64 x 64
            DenseBlockEncoder(ndf, 6),
            DenseTransitionBlockEncoder(ndf, ndf * 2, 2, activation=activation, args=args),

            # state size. (ndf*2) x 32 x 32
            DenseBlockEncoder(ndf * 2, 12),
            DenseTransitionBlockEncoder(ndf * 2, ndf * 4, 2, activation=activation, args=args),

            # state size. (ndf*4) x 16 x 16
            DenseBlockEncoder(ndf * 4, 16),
            DenseTransitionBlockEncoder(ndf * 4, ndf * 8, 2, activation=activation, args=args),

            # state size. (ndf*4) x 8 x 8
            DenseBlockEncoder(ndf * 8, 16),
            DenseTransitionBlockEncoder(ndf * 8, ndf * 8, 2, activation=activation, args=args),
            
            # state size. (ndf*4) x 8 x 8
            DenseBlockEncoder(ndf * 8, 16),
            DenseTransitionBlockEncoder(ndf * 8, ndf * 8, 2, activation=activation, args=args),

            # state size. (ndf*8) x 4 x 4
            DenseBlockEncoder(ndf * 8, 16),
            DenseTransitionBlockEncoder(ndf * 8, ndim, 4, activation=activation, args=args),
            
            f_activation(*f_args),
        )

    def forward(self, input):
        input = add_coordConv_channels(input)
        output = self.main(input).view(-1, self.ndim)
        return output

class waspDenseDecoder256(nn.Module):
    def __init__(self, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh,
                 f_args=[]):
        super(waspDenseDecoder256, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf * 8, 16),
            DenseTransitionBlockDecoder(ngf * 8, ngf * 8),
            
            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf * 8, 16),
            DenseTransitionBlockDecoder(ngf * 8, ngf * 8),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf * 8, 16),
            DenseTransitionBlockDecoder(ngf * 8, ngf * 4),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf * 4, 12),
            DenseTransitionBlockDecoder(ngf * 4, ngf * 2),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf * 2, 6),
            DenseTransitionBlockDecoder(ngf * 2, ngf),

            # state size. (ngf) x 64 x 64
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 128 x 128
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )

    def forward(self, inputs):
        return self.main(inputs)

class Backwardmapper(pl.LightningModule):
    #in_channels -> nc      | encoder first layer
    #filters -> ndf    | encoder first layer
    #img_size(h,w) -> ndim
    #out_channels  -> optical flow (x,y)

    def __init__(self, img_size=256, in_channels=3, out_channels=2, filters=32,fc_units=100, lr = 1e-3, weight_decay=5e-4):
        super(Backwardmapper, self).__init__()
        self.nc=in_channels
        self.nf=filters
        self.ndim=img_size
        self.oc=out_channels
        self.fcu=fc_units
        self.lr = lr
        self.weight_decay = weight_decay
        self.L1_loss = nn.L1Loss(reduction='none')
        #self.tensorboard = self.logger.experiment
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.encoder=waspDenseEncoder256(nc=self.nc+2,ndf=self.nf,ndim=self.ndim)
        self.decoder=waspDenseDecoder256(nz=self.ndim,nc=self.oc,ngf=self.nf)
        #self.decoder=waspDenseDecoder128(nz=self.ndim,nc=self.oc,ngf=64)
        # self.fc_layers= nn.Sequential(nn.Linear(self.ndim, self.fcu),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               nn.Linear(self.fcu,self.ndim),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               )

    def forward(self, inputs):
        #if inputs.shape[0] == 1:
        #    self.eval()
        encoded=self.encoder(inputs)
        #self.log("shape", str(encoded.shape))
        encoded=encoded.unsqueeze(-1).unsqueeze(-1)
        decoded=self.decoder(encoded)
        # print torch.max(decoded)
        # print torch.min(decoded)
        # für eine batch size von 1 funktioniert die batch norm nicht,
        # model.eval() muss dafür vorher ausgeführt werden. Ansonten die InstanceNorm nutzen
        #self.train()
        return decoded

    
    def test_forward(self, inputs):
        encoded=self.encoder(inputs)
        return encoded , self.decoder(encoded)
        #return encoded , encoded.unsqueeze(-1).unsqueeze(-1) , self.decoder(encoded.unsqueeze(-1).unsqueeze(-1))

    def l_angle_def(self, theta_x, theta_y, theta_x_gt, theta_y_gt , type = 'paper'):
        if type == 'paper':
            l_x = (torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi) % (2*math.pi)
            l_y = (torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi) % (2*math.pi)
            l_angle = torch.add(l_x, l_y)
            return l_angle
        else:
            l_x = (math.pi) - torch.abs((torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi))
            l_y = (math.pi) - torch.abs((torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi))
            l_angle = torch.add(l_x, l_y)
            return l_angle

    def loss_calculation(self, inputs, labels, log_type = 'train'):
        encoded=self.encoder(inputs)
        encoded=encoded.unsqueeze(-1).unsqueeze(-1)
        decoded=self.decoder(encoded)

        #l1_loss = self.L1_loss(decoded,labels['warped_bm'])
        l1_loss = torch.norm((decoded - labels['warped_bm']),p=1,dim=(1))
        l1_loss = l1_loss.unsqueeze(1)


        # angle loss 
        angles_map = angles.calc_angles_torch(decoded.transpose(1,2).transpose(2,3))
        warped_angle = angles.warp_grid_torch(angles_map, labels['warped_uv'].transpose(1,2).transpose(2,3))
        warped_angle = warped_angle.transpose(3,2).transpose(2,1)
        theta_x = warped_angle[:,0:1,:,:]
        theta_y = warped_angle[:,1:2,:,:]

        theta_x_gt = labels['warped_angle'][:,0:1,:,:]
        theta_y_gt = labels['warped_angle'][:,1:2,:,:]
        l_angle = self.l_angle_def(theta_x, theta_y, theta_x_gt, theta_y_gt, 'test')
        l_angle = l_angle * labels['warped_text_mask']
        

        msssim_metric = ms_ssim( decoded, labels['warped_bm'], data_range=1, size_average=True)
        self.log('ms_ssim_' + log_type, msssim_metric, on_step=False, on_epoch=True)


        epe= torch.mean(torch.norm((decoded - labels['warped_bm']),p=1,dim=(1)))
        self.log('epe_' + log_type,epe, on_step=False, on_epoch=True)

        loss = torch.mean(l1_loss + l_angle)
        return loss

    def log_images(self, inputs, decoded, labels, log_type):
        for i in range(inputs.shape[0]):        
            tensorboard = self.logger.experiment

            unwarped_image = utils.unwarp_image_logging(labels['img'][i].unsqueeze(0),decoded[i].unsqueeze(0))
            tensorboard.add_image('Unwarped_image_' + str(i),unwarped_image, self.global_step)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels, log_type = 'validation')
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels, log_type = 'test')
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': sched,
            'monitor': 'validation_loss',
            }
        }
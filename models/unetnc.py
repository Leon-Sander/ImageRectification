import torch
import torch.nn as nn
from torch.nn import init
import functools
import pytorch_lightning as pl
import math

class Estimator3d(pl.LightningModule):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 lr = "1e-3", weight_decay=5e-4):
        super(Estimator3d, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

    def l_angle_def(self, theta_x, theta_y, theta_x_gt, theta_y_gt , type = 'paper'):
        if type == 'paper':
            l_x = (torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi) % (2*math.pi)
            l_y = (torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi) % (2*math.pi)
            l_angle = torch.add(l_x, l_y)
            return l_angle
        else:
            l_x = (math.pi) - torch.abs((torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi))
            l_y = (math.pi) - torch.abs((torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi))
            #l_x = (math.pi) - torch.abs((torch.norm(torch.sub(theta_x, theta_x_gt),p=2,dim=(1)) - math.pi))
            #l_y = (math.pi) - torch.abs((torch.norm(torch.sub(theta_y, theta_y_gt),p=2,dim=(1)) - math.pi))
            l_angle = torch.add(l_x, l_y)
            return l_angle
            

    def loss_calculation(self, outputs, labels):
        wc_coordinates = outputs[:,0:3,:,:]
        l1_loss = torch.norm((wc_coordinates - labels['wc_gt']),p=1,dim=(1))

        
        phi_xx = outputs[:,3:4,:,:]
        phi_xy = outputs[:,4:5,:,:]
        phi_yx = outputs[:,5:6,:,:]
        phi_yy = outputs[:,6:7,:,:]
        curvature_mesh = outputs[:,7:8,:,:]

        

        theta_x = torch.atan2(phi_xx, phi_xy)
        theta_y = torch.atan2(phi_yx, phi_yy)
        #p_x = torch.norm(phi_xx, phi_xy,p=2,dim=(1,2,3)) Bei der Norm darf nur ein Wert eingegeben Werden
        #p_y = torch.norm(phi_yx, phi_yy,p=2,dim=(1,2,3))
        theta_x_gt = labels['warped_angle_gt'][:,0:1,:,:]
        theta_y_gt = labels['warped_angle_gt'][:,1:2,:,:]
        l_angle = self.l_angle_def(theta_x, theta_y, theta_x_gt, theta_y_gt, 'test')
        l_angle = l_angle * labels['warped_text_mask']

        
        l_curvature = torch.norm((curvature_mesh - labels['warped_curvature_gt']),p=2,dim=(1))
        
        loss = l1_loss + l_angle + l_curvature
        loss = torch.mean(loss)

        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        loss = self.loss_calculation(outputs, labels)
        self.log("loss", loss)
        return loss
 
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)

        loss = self.loss_calculation(outputs, labels)
        self.log("val_loss", loss, on_epoch=True) 
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)

        loss = self.loss_calculation(outputs, labels)
        self.log("test_loss", loss, on_epoch=True) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': sched,
            'monitor': 'loss',
            }
        }#'''

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
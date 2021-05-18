import torch
import torch.nn as nn
from torch.nn import init
import functools
import pytorch_lightning as pl

class Estimator3d(pl.LightningModule):
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

    def split_tensors(self, model_output):
        """Method to split the output channels of the prediction, to further calculate the loss


        :param model_output: prediction of the model 
        :type model_output: [type]
        """
        batch_size = model_output.shape[0]
        wc_coordinates_pred = torch.empty(batch_size,3,256,256)
        #angle_pred = torch.empty(batch_size,4,256,256)
        phi_xx = torch.empty(batch_size,1,256,256)
        phi_xy = torch.empty(batch_size,1,256,256)
        phi_yx = torch.empty(batch_size,1,256,256)
        phi_yy = torch.empty(batch_size,1,256,256)

        idx = 0
        for output in model_output:
            
            wc_coordinates_pred[idx] = output[:3]
            #angle_pred[idx] = output[3:]
            phi_xx[idx] = output[3]
            phi_xy[idx] = output[4]
            phi_yx[idx] = output[5]
            phi_yy[idx] = output[6]

            idx += 1

        angle_pred = {'phi_xx' : phi_xx, 'phi_xy' : phi_xy, 'phi_yx' : phi_yx, 'phi_yy' : phi_yy,}
        return wc_coordinates_pred, angle_pred

    def training_step(self, batch, batch_idx):
        images, labels = batch

        htan = nn.Hardtanh(0,1.0)
        MSE = nn.MSELoss()
        loss_fn = nn.L1Loss()

        outputs = self.model(images)
        wc_coordinates_output, angle_output = self.split_tensors(outputs)

        # l1 loss
        wc_coordinates_pred=htan(wc_coordinates_output)
        l1_loss = loss_fn(wc_coordinates_pred, labels['wc_gt'])

        # loss of angle prediction
        theta_x = torch.atan2(angle_output['phi_xx'], angle_output['phi_xy'])
        theta_y = torch.atan2(angle_output['phi_yx'], angle_output['phi_yy'])
        p_x = torch.cdist(angle_output['phi_xx'],angle_output['phi_xy'])
        p_y = torch.cdist(angle_output['phi_yx'],angle_output['phi_yy'])

        C = wc_coordinates_output - labels['wc_gt']
        angle_loss = torch.norm(C,p=1)
        
         
        
        #loss=l1loss



        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=True)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': sched,
            'monitor': 'metric_to_track',
            }
        }'''

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
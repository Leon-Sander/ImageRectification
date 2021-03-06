import torch
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl
from models.unetnc import Estimator3d
from models.backwardmapper import Backwardmapper
import angles
import math
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from icecream import ic
import torch.nn as nn
import sys


class crease(pl.LightningModule):
    def __init__(self, num_downs = 5, input_nc_wc = 3, output_nc_wc = 8, img_size = 256 , use_pre_trained = False, ngf_wc=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, lr = 1e-3, weight_decay=5e-4,
                 load_3d = 'estimator3d', load_bm = 'backward_map_estimator', angle_loss_type = 'ours'): #img_size
        super(crease, self).__init__()

        self.L1_loss = nn.L1Loss(reduction='none')
        self.lr = lr
        self.weight_decay = weight_decay
        self.angle_loss_type = angle_loss_type

        self.estimator3d = Estimator3d(input_nc = 3, output_nc = 8, num_downs = 5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.backward_map_estimator = Backwardmapper(img_size, in_channels=3, out_channels=2, filters=32,fc_units=100)
        if use_pre_trained:
            self.estimator3d.load_state_dict(torch.load('models/pretrained/' + load_3d + '.pkl'))
            self.backward_map_estimator.load_state_dict(torch.load('models/pretrained/' + load_bm + '.pkl'))
        


    def forward(self, input):
        warped_wc = self.estimator3d.forward(input)
        output = self.backward_map_estimator(warped_wc[:,0:3,:,:])
        #unwarped_img = self.unwarp_image(input, output.transpose(1,2).transpose(2,3)) 
        return output


    def l_angle_def(self, theta_x, theta_y, theta_x_gt, theta_y_gt , type = 'paper', p_x = 0, p_y = 0, bm_wc = 'bm'):
        if type == 'paper':
            if bm_wc == 'wc':
                l_x = ((torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi)* p_x) % (2*math.pi)
                l_y = ((torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi)* p_y) % (2*math.pi)
                l_angle = torch.add(l_x, l_y)
                return l_angle
            else:
                l_x = (torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi) % (2*math.pi)
                l_y = (torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi) % (2*math.pi)
                l_angle = torch.add(l_x, l_y)
                return l_angle 
        else:
            if bm_wc == 'wc':
                l_x = ((math.pi) - torch.abs((torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi))) * p_x
                l_y = ((math.pi) - torch.abs((torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi))) * p_y
                l_angle = torch.add(l_x, l_y)
                return l_angle
            else:
                l_x = (math.pi) - torch.abs((torch.abs(torch.sub(theta_x, theta_x_gt)) - math.pi))
                l_y = (math.pi) - torch.abs((torch.abs(torch.sub(theta_y, theta_y_gt)) - math.pi))
                l_angle = torch.add(l_x, l_y)
                return l_angle
            

    def l1_loss_calculation(self, outputs, labels):
        wc_coordinates = outputs[:,0:3,:,:]
        l1_loss = torch.norm((wc_coordinates - labels['wc_gt']),p=1,dim=(1))
        #l1_loss = self.L1_loss(wc_coordinates,labels['wc_gt'])
        l1_loss = l1_loss.unsqueeze(1)
        #print(l1_loss.shape)
        
        phi_xx = outputs[:,3:4,:,:]
        phi_xy = outputs[:,4:5,:,:]
        phi_yx = outputs[:,5:6,:,:]
        phi_yy = outputs[:,6:7,:,:]
        curvature_mesh = outputs[:,7:8,:,:]

        

        theta_x = torch.atan2(phi_xx, phi_xy)
        theta_y = torch.atan2(phi_yx, phi_yy)

        p_x = torch.norm((phi_xx - phi_xy),p=2,dim=(1)) #Bei der Norm darf nur ein Wert eingegeben Werden
        p_y = torch.norm((phi_yx - phi_yy),p=2,dim=(1))
        p_x = p_x.unsqueeze(1)
        p_y = p_y.unsqueeze(1)

        theta_x_gt = labels['warped_angle'][:,0:1,:,:]
        theta_y_gt = labels['warped_angle'][:,1:2,:,:]

        l_angle = self.l_angle_def(theta_x, theta_y, theta_x_gt, theta_y_gt, self.angle_loss_type, p_x, p_y, 'wc')
        l_angle = l_angle * labels['warped_text_mask']
        #print(l_angle.shape)

        
        l_curvature = torch.norm((curvature_mesh- labels['warped_curvature_gt']),p=2,dim=(1))
        l_curvature = l_curvature.unsqueeze(1)
        #print(l_curvature.shape)
        #ic(l1_loss.shape, l_angle.shape, l_curvature.shape)
        loss = l1_loss + l_angle + l_curvature
        #ic(loss.shape)
        #loss = torch.mean(loss)
        #print(loss.shape)
        return loss

    def loss_calculation(self, images, labels, log_type = 'train'):
        wc_coordinates = self.estimator3d(images)
        l3d_loss = self.l1_loss_calculation(wc_coordinates, labels)

        backward_map = self.backward_map_estimator(wc_coordinates[:,0:3,:,:])

        
        bm_loss = torch.norm((backward_map - labels['warped_bm']),p=1,dim=(1))
        bm_loss = bm_loss.unsqueeze(1)
        #bm_loss = self.L1_loss(backward_map,labels['warped_bm'])
        

        unwarped_img = self.unwarp_image(images, backward_map.transpose(1,2).transpose(2,3))
        unwarped_img_gt = self.unwarp_image(images, labels['warped_bm'].transpose(1,2).transpose(2,3))
        
        #metric logging
        epe= torch.mean(torch.norm((unwarped_img - unwarped_img_gt),p=1,dim=(1)))
        self.log('epe_' + log_type,epe, on_step=False, on_epoch=True)
        msssim_metric = ms_ssim( unwarped_img, unwarped_img_gt, data_range=1, size_average=True)
        self.log('ms_ssim_' + log_type, msssim_metric, on_step=False, on_epoch=True)
        #self.log_images(unwarped_img)

        angles_map = angles.calc_angles_torch(backward_map.transpose(1,2).transpose(2,3))
        warped_angle = angles.warp_grid_torch(angles_map, labels['warped_uv'].transpose(1,2).transpose(2,3))
        warped_angle = warped_angle.transpose(3,2).transpose(2,1)
        theta_x = warped_angle[:,0:1,:,:]
        theta_y = warped_angle[:,1:2,:,:]

        theta_x_gt = labels['warped_angle'][:,0:1,:,:]
        theta_y_gt = labels['warped_angle'][:,1:2,:,:]
        l_angle = self.l_angle_def(theta_x, theta_y, theta_x_gt, theta_y_gt, self.angle_loss_type)
        #l_angle = l_angle * labels['warped_text_mask']

        #ic(l3d_loss.shape, bm_loss.shape, l_angle.shape)
        #sys.exit(1)
        loss = l3d_loss + bm_loss + l_angle
        loss = torch.mean(loss)
        return loss
    

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels, log_type= 'validation')
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        loss = self.loss_calculation(inputs, labels, log_type= 'test')
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': sched,
            'monitor': 'validation_loss',
            }
        }
    
    def log_images(self, unwarped_images, log_type):
        '''
        logs per step the first unwarped image
        '''        
        tensorboard = self.logger.experiment
        tensorboard.add_image('Unwarped_image',unwarped_images[0], self.global_step)

    def unwarp_image(self, img, bm):
        assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
        
        n, c, h, w = img.shape

        bm = bm.transpose(3, 2).transpose(2, 1)
        bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
        bm = bm.transpose(1, 2).transpose(2, 3)

        #bm = 2 * bm - 1 # adapt value range for grid_sample
        bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
        
        img = img.float()
        res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
        res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
        return res
        
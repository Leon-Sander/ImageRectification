
import torch.nn.functional as F
import numpy as np
import torch
import cv2
from models.full_model import crease
import matplotlib.pyplot as plt
from models.backwardmapper import Backwardmapper

crease_model = crease()
crease_model.load_state_dict(torch.load('models/pretrained/crease_test.pkl'))

bm_model = Backwardmapper()
bm_model.load_state_dict(torch.load('models/pretrained/bm_test.pkl'))

def plt_result_crease(path):

    img = load_img(path)
    bm = crease_model(img)
    unwarped_image_pred = unwarp_image(img,bm)
    unwarped_image_gt = unwarp_image(img,load_bm(path).unsqueeze(0))


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Model Prediction')

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    plt.axis('off')


def plt_result_bm(path):

    img = load_img(path)
    wc = load_wc(path)

    bm = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image(img,bm)
    unwarped_image_gt = unwarp_image(img,load_bm(path).unsqueeze(0))


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Model Prediction')

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    plt.axis('off')

def load_wc(path):
    wc = np.load(path + '/warped_WC.npz')['warped_WC']
    lbl = wc
    msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
    #xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
    xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 # preview
    lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
    lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
    lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
    lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
    #lbl = cv2.resize(lbl, 256, interpolation=cv2.INTER_NEAREST)
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()


    return lbl


def load_bm(path):
    bm = np.load(path + '/warped_BM.npz')['warped_BM']
    lbl = bm
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()
    return lbl

def load_img(path):
    img = cv2.imread(path + '/warped_document.png')
    #img = dataset_train.transform_img(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)#.transpose(2,3).transpose
    img = img / 255
    return img

#@staticmethod
def unwarp_image(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    
    n, c, h, w = img.shape

    #bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    res = res.transpose(1,2).transpose(2,3).detach().numpy()[0]
    return res
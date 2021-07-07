
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import cv2
#from models.full_model import crease
import matplotlib.pyplot as plt
#from models.backwardmapper import Backwardmapper
from pytorch_ssim_.pytorch_ssim import SSIM
from icecream import ic
import random

ssim = SSIM()

#crease_model = crease()
#crease_model.load_state_dict(torch.load('models/pretrained/crease_test.pkl'))

#bm_model = Backwardmapper()
#bm_model.load_state_dict(torch.load('models/pretrained/bm_test2.pkl'))



def compare_l1(path, bm_model):
    L1_loss = nn.L1Loss(reduction='none')
    wc = load_wc(path)
    bm_gt = load_bm(path).unsqueeze(0)
    bm = bm_model(wc.unsqueeze(0))

    l1_loss = L1_loss(bm,bm_gt)
    return l1_loss


def unwarp_and_ssim(bm, bm_gt, img):
    #ic(bm.shape)
    #ic(bm_gt.shape)
    #ic(img.shape)
    unwarped_image_pred = unwarp_image_ssmi(img,bm)
    unwarped_image_gt = unwarp_image_ssmi(img,bm_gt)


    #ic(unwarped_image_pred.shape)
    #ic(unwarped_image_gt.shape)
    return ssim.forward(unwarped_image_pred, unwarped_image_gt)


def compare_ssim_all(path, bm_model):
    

    img = load_warped_document(path)
    wc = load_wc(path)

    bm = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image_ssmi(img,bm)
    unwarped_image_gt = unwarp_image_ssmi(img,load_bm(path).unsqueeze(0))

    print(ssim(unwarped_image_pred, unwarped_image_gt))



def plt_result_crease(path, crease_model):

    img = load_warped_document(path)
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


def plt_result_bm(path, bm_model):

    img = load_warped_document(path)
    wc = load_wc(path)

    bm = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image(img,bm)
    unwarped_image_gt = unwarp_image(img,load_bm(path).unsqueeze(0))

    unwarped_image_pred_ssmi = unwarp_image_ssmi(img,bm)
    unwarped_image_gt_ssmi = unwarp_image_ssmi(img,load_bm(path).unsqueeze(0))

    ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))

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

def load_uv(path):
    lbl = np.load(path + '/warped_UV.npz')['warped_UV']
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()
    return lbl

def load_anlges(path):
    lbl = np.load(path + '/warped_angle.npz')['warped_angle']
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()
    return lbl

def load_txt_msk(path):
    lbl = np.load(path + '/warped_text_mask.npz')['warped_text_mask']
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

def load_warped_document(path):
    #### img gets loaded with shape (n,c,h,w)
    img = cv2.imread(path + '/warped_document.png')
    #img = dataset_train.transform_img(img)
    img = img.transpose(2, 0, 1)
    
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)#.transpose(2,3).transpose
    img = img / 255
    return img


def load_warped_document_chw(path):
    #### img gets loaded with shape (n,c,h,w)
    img = cv2.imread(path + '/warped_document.png')
    #img = dataset_train.transform_img(img)
    img = img.transpose(2, 0, 1)
    
    img = torch.from_numpy(img).float()
    img = img / 255
    return img

#@staticmethod
def unwarp_image(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    #ic(img.shape)
    #ic(bm.shape)
    n, c, h, w = img.shape

    #bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    res = res.transpose(1,2).transpose(2,3)#.detach()
    #ic(res)
    res = res.detach().cpu()
    res = res.numpy()[0]
    return res

def unwarp_image_ssmi(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    #ic(img.shape)
    #ic(bm.shape)
    n, c, h, w = img.shape

    #bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    #res.unsqueeze(0)
    #res = res.transpose(1,2).transpose(2,3)#.detach()
    #ic(res)
    res = res.detach().cpu()
    #res = res.numpy()#[0]
    #res = np.array(res, dtype=np.float64)
    #ic(res.shape)
    return res

def unwarp_image_logging(img, bm):
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
    #res = res.transpose(1,2).transpose(2,3).detach()
    res = res.detach().cpu().numpy()[0]
    return res

def tight_crop_wc_bm( wc, bm):
    msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    wc = wc[miny : maxy + 1, minx : maxx + 1, :]

    
    #s = 20
    #wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')

    #cx1 = random.randint(0, s - 5)
    #cx2 = random.randint(0, s - 5) + 1
    #cy1 = random.randint(0, s - 5)
    #cy2 = random.randint(0, s - 5) + 1

    #wc = wc[cy1 : -cy2, cx1 : -cx2, :]

    #t=miny-s+cy1
    #b=size[0]-maxy-s+cy2
    #l=minx-s+cx1
    #r=size[1]-maxx-s+cx2

    t=miny
    b=size[0]-maxy
    l=minx
    r=size[1]-maxx

    img_size = (256,256)


    msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
    #normalize label
    xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 
    wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
    wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
    wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
    wc=cv2.bitwise_and(wc,wc,mask=msk)
    
    wc = cv2.resize(wc, img_size) 
    wc = wc.astype(float) / 255.0
    wc = wc.transpose(2, 0, 1) # NHWC -> NCHW
    wc = np.array(wc, dtype=np.float64)
    wc = torch.from_numpy(wc).float()

    bm = bm.astype(float)
    #normalize label [-1,1]
    bm[:,:,1]=bm[:,:,1]-t
    bm[:,:,0]=bm[:,:,0]-l
    bm=bm/np.array([256.0-l-r, 256.0-t-b])
    bm=(bm-0.5)*2

    bm0=cv2.resize(bm[:,:,0],(img_size[0],img_size[1]))
    bm1=cv2.resize(bm[:,:,1],(img_size[0],img_size[1]))
    
    #img=np.concatenate([alb,wc],axis=0)
    bm=np.stack([bm0,bm1],axis=-1)

    #img = torch.from_numpy(img).float()
    bm = torch.from_numpy(bm).float()



    return wc, bm


def unwarp_image_crop(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    #ic(img.shape)
    #ic(bm.shape)
    n, c, h, w = img.shape

    bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    res = res.transpose(1,2).transpose(2,3)#.detach()
    #ic(res)
    res = res.detach().cpu()
    res = res.numpy()[0]
    return res

def tight_crop_im_wc(im, fm):
    # different tight crop
    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]
    
    # px = int((maxx - minx) * 0.07)
    # py = int((maxy - miny) * 0.07)
    
    # im = np.pad(im, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # fm = np.pad(fm, ((py, py + 1), (px, px + 1), (0, 0)), 'constant')
    # # crop
    # cx1 = int(random.randint(0, 3) / 7.0 * px)
    # cx2 = int(random.randint(0, 3) / 7.0 * px + 1)
    # cy1 = int(random.randint(0, 3) / 7.0 * py)
    # cy2 = int(random.randint(0, 3) / 7.0 * py + 1)
    
    #s = 20
    #im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    #fm = np.pad(fm, ((s, s), (s, s), (0, 0)), 'constant')
    #cx1 = random.randint(0, s - 5)
    #cx2 = random.randint(0, s - 5) + 1
    #cy1 = random.randint(0, s - 5)
    #cy2 = random.randint(0, s - 5) + 1

    #im = im[cy1 : -cy2, cx1 : -cx2, :]
    #fm = fm[cy1 : -cy2, cx1 : -cx2, :]
    return im, fm

def tight_crop_im_bm(im, fm, bm):
    img_size = (256,256)
    ######## preprocessing
    wc = fm
    msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
    #normalize label
    xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 
    wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
    wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
    wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
    wc=cv2.bitwise_and(wc,wc,mask=msk)
    
    wc = cv2.resize(wc, img_size) 
    wc = wc.astype(float) / 255.0
    #wc = wc.transpose(2, 0, 1) # NHWC -> NCHW
    #wc = np.array(wc, dtype=np.float64)
    #wc = torch.from_numpy(wc).float()

    #im = im / 255

    
    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
 
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]


    t=miny
    b=size[0]-maxy
    l=minx
    r=size[1]-maxx

    ic(t)
    ic(b)
    ic(l)
    ic(r)

    
    bm = np.array(bm, dtype=np.float64)
    #bm = bm.astype(float)
    #normalize label [-1,1]
    bm = bm*255
    bm[:,:,1]=bm[:,:,1]-t
    bm[:,:,0]=bm[:,:,0]-l
    ic(bm.shape)
    bm=bm/np.array([256.0-l-r, 256.0-t-b])
    #bm=bm/np.array([1.0-l-r, 1.0-t-b])
    #bm=(bm-0.5)*2

    #bm0=cv2.resize(bm[:,:,0],(img_size[0],img_size[1]))
    #bm1=cv2.resize(bm[:,:,1],(img_size[0],img_size[1]))
    
    #img=np.concatenate([alb,wc],axis=0)
    #bm=np.stack([bm0,bm1],axis=-1)

    #img = torch.from_numpy(img).float()
    bm = torch.from_numpy(bm).float()

    img = cv2.resize(im.numpy(), (256,256), interpolation=cv2.INTER_NEAREST)

    #img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    #img = img / 255

    return img, fm, bm

def tight_crop_no_fm(im, bm):
    #im = im*255
    img_size = (256,256)
    ######## preprocessing
    
    msk=((im[:,:,0]!=0)&(im[:,:,1]!=0)&(im[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
 
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]



    t=miny
    b=size[0]-maxy
    l=minx
    r=size[1]-maxx

    ic(t)
    ic(b)
    ic(l)
    ic(r)


    
    bm = np.array(bm, dtype=np.float64)
    #bm = bm.astype(float)
    #normalize label [-1,1]
    bm = bm*255
    bm[:,:,1]=bm[:,:,1]-t
    bm[:,:,0]=bm[:,:,0]-l
    ic(bm.shape)
    bm=bm/np.array([256.0-l-r, 256.0-t-b])
    #bm=bm/np.array([1.0-l-r, 1.0-t-b])
    #bm=(bm-0.5)*2

    #bm0=cv2.resize(bm[:,:,0],(img_size[0],img_size[1]))
    #bm1=cv2.resize(bm[:,:,1],(img_size[0],img_size[1]))
    
    #img=np.concatenate([alb,wc],axis=0)
    #bm=np.stack([bm0,bm1],axis=-1)

    #img = torch.from_numpy(img).float()
    bm = torch.from_numpy(bm).float()

    img = cv2.resize(im, (256,256), interpolation=cv2.INTER_NEAREST)

    #img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    #img = img / 255

    return img, bm

def load_warped_document_cropped(path):
    #### img gets loaded with shape (n,c,h,w)
    img = cv2.imread(path + '/warped_document.png')
    #img = dataset_train.transform_img(img)



    fm = np.load(path + '/warped_WC.npz')['warped_WC']
    # different tight crop
    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    img = img[miny : maxy + 1, minx : maxx + 1, :]
    ic(img.shape)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
        
    img = img.transpose(2, 0, 1)
    
    
    img = img / 255
    #img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)


    return img
    

def plt_result_bm_cropped(path, bm_model):

    img = load_warped_document(path)
    img_cropped = load_warped_document_cropped(path)
    wc = load_wc(path)

    bm = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image(img_cropped,bm)
    unwarped_image_gt = unwarp_image(img,load_bm(path).unsqueeze(0))

    unwarped_image_pred_ssmi = unwarp_image_ssmi(img,bm)
    unwarped_image_gt_ssmi = unwarp_image_ssmi(img,load_bm(path).unsqueeze(0))

    ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    
    plt.axis('off')
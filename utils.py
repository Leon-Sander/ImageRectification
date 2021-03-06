
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import cv2
#from models.full_model import crease
import matplotlib.pyplot as plt
#from models.backwardmapper import Backwardmapper
from icecream import ic
import random
from pytorch_msssim import ms_ssim, SSIM, MS_SSIM



def compare_l1(path, bm_model):
    L1_loss = nn.L1Loss(reduction='none')
    wc = load_wc(path)
    bm_gt = load_bm(path).unsqueeze(0)
    bm = bm_model(wc.unsqueeze(0))

    l1_loss = L1_loss(bm,bm_gt)
    return l1_loss


def unwarp_and_ssim(bm, bm_gt, img):
    unwarped_image_pred = unwarp_image_ssim(img,bm)
    unwarped_image_gt = unwarp_image_ssim(img,bm_gt)

    return ms_ssim(unwarped_image_pred, unwarped_image_gt, data_range=1, size_average=True)


def compare_ssim_all(path, bm_model):
    img = load_warped_document(path)
    wc = load_wc(path)

    bm = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image_ssim(img,bm)
    unwarped_image_gt = unwarp_image_ssim(img,load_bm(path).unsqueeze(0))

    print(ms_ssim(unwarped_image_pred, unwarped_image_gt, data_range=1, size_average=True))



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

    unwarped_image_pred_ssim = unwarp_image_ssim(img,bm)
    unwarped_image_gt_ssim = unwarp_image_ssim(img,load_bm(path).unsqueeze(0))


    ssim_metric = ms_ssim(unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)

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

    #bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    res = res.transpose(1,2).transpose(2,3)#.detach()
    #ic(res)
    res = res.detach().cpu()
    res = res.numpy()[0]
    return res

def unwarp_image_ssim(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    #ic(img.shape)
    #ic(bm.shape)
    n, c, h, w = img.shape

    #bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    #bm = 2 * bm - 1 # adapt value range for grid_sample
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

    #bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    #res = res.transpose(1,2).transpose(2,3).detach()
    res = res.detach().cpu().numpy()[0]
    return res



def unwarp_image_crop(img, bm):
    #assert bm.shape[3] == 2, "BM shape needs to be (N, H, W, C)"
    #ic(img.shape)
    #ic(bm.shape)
    n, c, h, w = img.shape

    bm = bm.transpose(3, 2).transpose(2, 1)
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True) # align_corners=True -> old behaviour
    bm = bm.transpose(1, 2).transpose(2, 3)

    #bm = 2 * bm - 1 # adapt value range for grid_sample
    bm = bm.transpose(1, 2) # rotate image by 90 degrees (NOTE: this transformation might be deleted in future BM versions)
    
    img = img.float()
    res = F.grid_sample(input=img, grid=bm, align_corners=True) # align_corners=True -> old behaviour
    res = torch.clamp(res, 0, 1) # clip values because of numerical instabilities
    res = res.transpose(1,2).transpose(2,3)#.detach()
    #ic(res)
    res = res.detach().cpu()
    res = res.numpy()[0]
    return res


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

    unwarped_image_pred_ssim = unwarp_image_ssim(img,bm)
    unwarped_image_gt_ssim = unwarp_image_ssim(img,load_bm(path).unsqueeze(0))

    #ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)
    ssim_metric = ms_ssim(unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    
    plt.axis('off')

def load_wc_cropped(path):
    fm = np.load(path + '/warped_WC.npz')['warped_WC']

    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()

    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    
    wc = fm[miny : maxy + 1, minx : maxx + 1, :]
    wc = cv2.resize(wc, (256,256), interpolation=cv2.INTER_NEAREST)

    lbl = wc
    msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
    #xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
    xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 # preview -> neu ausrechnen?
    lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
    lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
    lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
    lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
    lbl = cv2.resize(lbl, (256,256), interpolation=cv2.INTER_NEAREST)
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()


    return lbl


def plt_bm_gt_cropped(path, bm_model):
    #img = load_warped_document(path)
    wc, labels = crop_all(path)

    img_cropped = labels['img'].unsqueeze(0)

    #print(labels['warped_bm'].shape, labels['warped_bm'].max(), labels['warped_bm'].min())
    
    bm_pred = bm_model(wc.unsqueeze(0))
    unwarped_image_pred = unwarp_image(img_cropped,bm_pred)
    unwarped_image_gt = unwarp_image(img_cropped,labels['warped_bm'].unsqueeze(0))

    unwarped_image_pred_ssim = unwarp_image_ssim(img_cropped,bm_pred)
    unwarped_image_gt_ssim = unwarp_image_ssim(img_cropped,labels['warped_bm'].unsqueeze(0))


    msssim_metric = ms_ssim( unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img_cropped[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Prediction, ssmi to gt: ' + str(msssim_metric))

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    
    plt.axis('off')

def crop_all(data_path):
    img_size = (256,256)
    input = np.load(data_path + '/warped_WC.npz')['warped_WC']
    
    labels = {}
    labels['warped_bm'] = np.load(data_path + '/warped_BM.npz')['warped_BM']
    labels['warped_uv'] = np.load(data_path + '/warped_UV.npz')['warped_UV']
    labels['warped_angle'] = np.load(data_path + '/warped_angle.npz')['warped_angle']
    labels['warped_text_mask'] = np.load(data_path + '/warped_text_mask.npz')['warped_text_mask']
    labels['img'] = cv2.imread(data_path + '/warped_document.png')

    fm = input
    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()

    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]
    fm = cv2.resize(fm, img_size, interpolation=cv2.INTER_NEAREST)

    for label in labels:
        if label != 'warped_bm':
            im = labels[label]
            im = np.array(im, dtype=np.float64)
            #ic(label, im.shape)
            im = im[miny : maxy + 1, minx : maxx + 1, :]
            im = cv2.resize(im, img_size, interpolation=cv2.INTER_NEAREST)
            if label == 'warped_text_mask':
                im = np.expand_dims(im, axis=2)
            #ic(label, im.shape)
            labels[label] = im
        else:

            t=miny
            b=size[0]-maxy
            l=minx
            r=size[1]-maxx

            bm = labels[label]
            bm = np.array(bm, dtype=np.float64)

            #bm = bm*255
            bm = bm*448
            bm[:,:,1]=bm[:,:,1]-t
            bm[:,:,0]=bm[:,:,0]-l
            bm=bm/np.array([float(448)-l-r, float(448)-t-b])
            bm=(bm-0.5)*2
            bm = cv2.resize(bm, img_size, interpolation=cv2.INTER_NEAREST)
            labels[label] = bm

            #lbl = input

    lbl = fm
    msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
    xmx, xmn, ymx, ymn,zmx, zmn= 1.2361085,-1.2319995, 1.2294204, -1.210581, 0.5923838, -0.62981504   # calculate from all the wcs
    #xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 # preview
    lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
    lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
    lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
    lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
    lbl = cv2.resize(lbl, img_size, interpolation=cv2.INTER_NEAREST)
    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
    lbl = np.array(lbl, dtype=np.float64)
    lbl = torch.from_numpy(lbl).float()
    input = lbl
    #ic(input.shape)
    for label in labels:
        #ic(label, lbl.shape)
        if label == 'img':
            img = labels[label]
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            
            img = img / 255
            labels[label] = img
        else:

            lbl = labels[label]
            
            lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
            lbl = np.array(lbl, dtype=np.float64)
            lbl = torch.from_numpy(lbl).float()
            labels[label] = lbl

    return input, labels

def plt_bm_gt_cropped_2(bm_pred, img_cropped, bm):


    img_cropped = img_cropped.unsqueeze(0)
    #ic(img_cropped.shape)
    #ic(bm_pred.shape)
    unwarped_image_pred = unwarp_image(img_cropped,bm_pred)
    unwarped_image_gt = unwarp_image(img_cropped,bm.unsqueeze(0))

    unwarped_image_pred_ssim = unwarp_image_ssim(img_cropped,bm_pred)
    unwarped_image_gt_ssim = unwarp_image_ssim(img_cropped,bm.unsqueeze(0))

    #ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)
    ssim_metric = ms_ssim( unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img_cropped[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')

    ax2.imshow(unwarped_image_pred)
    ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Unwarped image gt')
    
    plt.axis('off')

def plt_crease(path, crease, counter):
    wc, labels = crop_all(path)

    img_cropped = labels['img'].unsqueeze(0)
    img_uncropped = load_warped_document_resized(path)
    

    bm_crease = crease(img_cropped)


    unwarped_image_pred = unwarp_image(img_cropped,bm_crease)
    unwarped_image_gt = unwarp_image(img_cropped,labels['warped_bm'].unsqueeze(0))

    unwarped_image_pred_ssim = unwarp_image_ssim(img_cropped,bm_crease)
    unwarped_image_gt_ssim = unwarp_image_ssim(img_cropped,labels['warped_bm'].unsqueeze(0))


    msssim_metric = ms_ssim(unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)
    #ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    #ax1.imshow(img_cropped[0].transpose(0,1).transpose(1,2))
    ax1.imshow(img_uncropped[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')
    ax1.axis('off')

    ax2.imshow(unwarped_image_pred)
    #ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))
    ax2.set_title('Prediction, MS-SSIM to GT: ' + str(msssim_metric))
    ax2.axis('off')

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Ground Truth')
    ax3.axis('off')


    #plt.savefig('imgs/' + str(counter) + '.png')
    #plt.close()
    

def load_warped_document_resized(path):
    #### img gets loaded with shape (n,c,h,w)
    img = cv2.imread(path + '/warped_document.png')
    #img = dataset_train.transform_img(img)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
    img = img.transpose(2, 0, 1)

    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)#.transpose(2,3).transpose
    img = img / 255
    return img

def plt_crease_774(path, crease):
    wc, labels = crop_all(path)

    img_cropped = labels['img'].unsqueeze(0)
    img_uncropped = load_warped_document_resized(path)
    

    bm_crease = crease(img_cropped)


    unwarped_image_pred = unwarp_image(img_cropped,bm_crease)
    unwarped_image_gt = unwarp_image(img_cropped,labels['warped_bm'].unsqueeze(0))

    unwarped_image_pred_ssmi = unwarp_image_ssim(img_cropped,bm_crease)
    unwarped_image_gt_ssmi = unwarp_image_ssim(img_cropped,labels['warped_bm'].unsqueeze(0))

    unwarped_image_pred = cv2.resize(unwarped_image_pred, (774,774), interpolation=cv2.INTER_NEAREST)
    unwarped_image_gt = cv2.resize(unwarped_image_gt, (774,774), interpolation=cv2.INTER_NEAREST)
    ic(unwarped_image_pred_ssmi.shape )
    ic(unwarped_image_pred_ssmi.numpy()[0].shape)
    unwarped_image_pred_ssmi  = cv2.resize(unwarped_image_pred_ssmi.numpy()[0] , (774,774), interpolation=cv2.INTER_NEAREST)
    unwarped_image_gt_ssmi = cv2.resize(unwarped_image_gt_ssmi.numpy()[0], (774,774), interpolation=cv2.INTER_NEAREST)

    unwarped_image_pred_ssmi = np.expand_dims(unwarped_image_pred_ssmi, 0)
    unwarped_image_gt_ssmi = np.expand_dims(unwarped_image_gt_ssmi, 0)

    msssim_metric = ms_ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi, data_range=1, size_average=True)
    #ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    #ax1.imshow(img_cropped[0].transpose(0,1).transpose(1,2))
    ax1.imshow(img_uncropped[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')
    plt.axis('off')
    ax2.imshow(unwarped_image_pred)
    #ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))
    ax2.set_title('Prediction, ms-ssmi to gt: ' + str(msssim_metric))
    plt.axis('off')
    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Ground Truth')
    
    plt.axis('off')


def calc_mssim(ordner, crease, data_dir):
    paths = []
    for item in ordner:
    #utils.compare_ssim_all(data_dir + item, bm_model)


        path = data_dir + 'test/' + item
        wc, labels = crop_all(path)

        img_cropped = labels['img'].unsqueeze(0)
        img_uncropped = load_warped_document_resized(path)
        

        bm_crease = crease(img_cropped)


        unwarped_image_pred = unwarp_image(img_cropped,bm_crease)
        unwarped_image_gt = unwarp_image(img_cropped,labels['warped_bm'].unsqueeze(0))

        unwarped_image_pred_ssmi = unwarp_image_ssim(img_cropped,bm_crease)
        unwarped_image_gt_ssmi = unwarp_image_ssim(img_cropped,labels['warped_bm'].unsqueeze(0))


        msssim_metric = ms_ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi, data_range=1, size_average=True)
        
        if msssim_metric.item() > 0.9:
            paths.append(path)


    return paths

def crease_save_fig(path, crease, counter):
    wc, labels = crop_all(path)

    img_cropped = labels['img'].unsqueeze(0)
    img_uncropped = load_warped_document_resized(path)
    

    bm_crease = crease(img_cropped)


    unwarped_image_pred = unwarp_image(img_cropped,bm_crease)
    unwarped_image_gt = unwarp_image(img_cropped,labels['warped_bm'].unsqueeze(0))

    unwarped_image_pred_ssim = unwarp_image_ssim(img_cropped,bm_crease)
    unwarped_image_gt_ssim = unwarp_image_ssim(img_cropped,labels['warped_bm'].unsqueeze(0))


    msssim_metric = ms_ssim(unwarped_image_pred_ssim, unwarped_image_gt_ssim, data_range=1, size_average=True)
    #ssim_metric = ssim(unwarped_image_pred_ssmi, unwarped_image_gt_ssmi)


    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    #ax1.imshow(img_cropped[0].transpose(0,1).transpose(1,2))
    ax1.imshow(img_uncropped[0].transpose(0,1).transpose(1,2))
    ax1.set_title('Warped Image')
    ax1.axis('off')

    ax2.imshow(unwarped_image_pred)
    #ax2.set_title('Prediction, ssmi to gt: ' + str(ssim_metric))
    ax2.set_title('Prediction, MS-SSIM to GT: ' + str(msssim_metric))
    ax2.axis('off')

    ax3.imshow(unwarped_image_gt)
    ax3.set_title('Ground Truth')
    ax3.axis('off')


    plt.savefig('imgs/' + str(counter) + '.png')
    plt.close()
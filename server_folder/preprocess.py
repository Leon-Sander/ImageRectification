import numpy as np
import cv2
import torch


def preprocess_img(img):
    img_size = (256,256)
    img = np.asarray(img)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST) # uint8 with RGB mode
    if img.shape[-1] == 4:
        img=img[:,:,:3]   # Discard the alpha channel  
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img).float()
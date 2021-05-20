import os
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import torch
from torchvision.io.image import ImageReadMode
from PIL import Image
from numpy import asarray
import cv2


class CustomImageDataset_wc(Dataset):
    """Dataset creation based on pytorch Dataset

    :param Dataset: Import from torch.utils.data.Dataset
    """
    def __init__(self, data_dir, transform=None, img_size = 256):
        #self.img_labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        # so wird auf den i-ten Ordner zugegriffen
        idx = sorted(os.listdir(self.data_dir))[idx]
        '''if idx < 10:
            data_path = os.path.join(self.data_dir, '0' + str(idx))
        else:
            data_path = os.path.join(self.data_dir, idx)'''
        data_path = os.path.join(self.data_dir, idx)
        #image = read_image(data_path + '/warped_document.png', mode = ImageReadMode.RGB) #->opening with torch image reader
        image =  Image.open(data_path + '/warped_document.png') #PIL image reader
        
        # opening all the ground truth files 
        labels = {}
        labels['wc_gt'] = np.load(data_path + '/warped_WC.npz')['warped_WC'] #world coordinates ground truth
        labels['warped_angle_gt'] = np.load(data_path + '/warped_angle.npz')['warped_angle']
        labels['warped_text_mask'] = np.load(data_path + '/warped_text_mask.npz')['warped_text_mask']
        labels['warped_curvature_gt'] = np.load(data_path + '/warped_curvature.npz')['warped_curvature']


        if self.transform:
            image = self.transform_img(image)
            labels = self.transform_labels(labels)

        #sample = {'image': image, 'label': labels}
        #return sample
        return image, labels


    def transform_img(self, img):
        # convert image to numpy array
        img = asarray(img)
        #img = m.imresize(img, self.img_size) # uint8 with RGB mode
        if img.shape[-1] == 4:
            img=img[:,:,:3]   # Discard the alpha channel  
        img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        img = torch.from_numpy(img).float()

        return img

    def transform_labels(self, labels):
        lbl = labels['wc_gt']
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        #lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float64)
        lbl = torch.from_numpy(lbl).float()
        labels['wc_gt'] = lbl

        for label in labels:
            if label != 'wc_gt':
                lbl = labels[label]
                lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
                lbl = np.array(lbl, dtype=np.float64)
                lbl = torch.from_numpy(lbl).float()
                labels[label] = lbl
                
        return labels
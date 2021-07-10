import os
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import torch
from torchvision.io.image import ImageReadMode
from PIL import Image
from numpy import asarray
import cv2
from pathlib import Path
from icecream import ic

class CustomImageDataset_wc(Dataset):
    """Dataset creation based on pytorch Dataset

    :param Dataset: Import from torch.utils.data.Dataset
    """
    def __init__(self, data_dir, transform=True, img_size = 256):
        #self.img_labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        #from pathlib import Path # pathlib Implementation probieren
        #p = Path('./')
        #list(p.glob('Dataset Preview/Inv3D preview complete/data/train/*/'))
        # so wird auf den i-ten Ordner zugegriffen
        idx = sorted(os.listdir(self.data_dir))[idx]
        '''if idx < 10:
            data_path = os.path.join(self.data_dir, '0' + str(idx))
        else:
            data_path = os.path.join(self.data_dir, idx)'''
        data_path = os.path.join(self.data_dir, idx)
        #image = read_image(data_path + '/warped_document.png', mode = ImageReadMode.RGB) #->opening with torch image reader
        image =  cv2.imread(data_path + '/warped_document.png') #PIL image reader
        
        # opening all the ground truth files 
        labels = {}
        labels['wc_gt'] = np.load(data_path + '/warped_WC.npz')['warped_WC'] #world coordinates ground truth
        labels['warped_angle_gt'] = np.load(data_path + '/warped_angle.npz')['warped_angle']
        labels['warped_text_mask'] = np.load(data_path + '/warped_text_mask.npz')['warped_text_mask']
        labels['warped_curvature_gt'] = np.load(data_path + '/warped_curvature.npz')['warped_curvature']


        if self.transform:
            image, labels = self.tight_crop_all(image, labels)
            image = self.transform_img(image)
            labels = self.transform_labels(labels)

        #sample = {'image': image, 'label': labels}
        #return sample
        return image, labels

    def tight_crop_all(self, img, labels):
        fm = labels['wc_gt']  
        msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
    
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        
        fm = fm[miny : maxy + 1, minx : maxx + 1, :]
        fm = cv2.resize(fm, self.img_size, interpolation=cv2.INTER_NEAREST)
        img = img[miny : maxy + 1, minx : maxx + 1, :]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
        labels['wc_gt'] = fm

        for label in labels:
            if label != 'wc_gt':
                im = labels[label]
                im = np.array(im, dtype=np.float64)
                #ic(label, im.shape)
                im = im[miny : maxy + 1, minx : maxx + 1, :]
                im = cv2.resize(im, self.img_size, interpolation=cv2.INTER_NEAREST)
                if label == 'warped_text_mask' or label == 'warped_curvature_gt':
                    im = np.expand_dims(im, axis=2)
                #ic(label, im.shape)
                labels[label] = im


        return img, labels


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
    
    @staticmethod
    def calculate_min_and_max(data_dir, img_size):
        
        #p = Path('./')
        #folder = list(p.glob(data_dir + '*/'))
        ordner = sorted(os.listdir(data_dir))
        folder = []
        for item in ordner:
            folder.append(data_dir + item )

        xmx, xmn, ymx, ymn, zmx, zmn = 0,0,0,0,0,0
        curv_mx = 0
        for item in folder:
            path_wc = str(item) + '/warped_WC.npz'
            wc_gt = np.load(path_wc)['warped_WC']

            fm = wc_gt
            msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
            size=msk.shape
            [y, x] = (msk).nonzero()
        
            minx = min(x)
            maxx = max(x)
            miny = min(y)
            maxy = max(y)
            
            fm = fm[miny : maxy + 1, minx : maxx + 1, :]
            fm = cv2.resize(fm, img_size, interpolation=cv2.INTER_NEAREST)
            wc_gt = fm

            wc_gt = wc_gt.transpose(2,0,1)

            local_zmx = np.amax(wc_gt[0])
            local_zmn = np.amin(wc_gt[0])
            local_ymx = np.amax(wc_gt[1])
            local_ymn = np.amin(wc_gt[1])
            local_xmx = np.amax(wc_gt[2])
            local_xmn = np.amin(wc_gt[2])

            if local_xmx > xmx:
                xmx = local_xmx
            if local_xmn < xmn:
                xmn = local_xmn

            if local_ymx > ymx:
                ymx = local_ymx
            if local_ymn < ymn:
                ymn = local_ymn

            if local_zmx > zmx:
                zmx = local_zmx
            if local_zmn < zmn:
                zmn = local_zmn

            path_curvature = str(item) + '/warped_curvature.npz'
            curv_gt = np.load(path_curvature)['warped_curvature']
            curv_gt = curv_gt[miny : maxy + 1, minx : maxx + 1, :]
            curv_gt = cv2.resize(curv_gt, img_size, interpolation=cv2.INTER_NEAREST)
            
            #curv_gt = curv_gt.unsqueeze(2)
            #ic(curv_gt.shape)
            #curv_gt = curv_gt.transpose(2,0,1)
            #ic(curv_gt.shape)
            curv_local_mx =  np.amax(curv_gt)
            if curv_local_mx > curv_mx:
                curv_mx = curv_local_mx

        return (xmx, xmn, ymx, ymn,zmx, zmn, curv_mx) 

    def transform_labels(self, labels):
        lbl = labels['wc_gt']
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        #xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        #xmx, xmn, ymx, ymn,zmx, zmn= 1.0858383, -1.0862498, 0.8847823, -0.8838696, 0.31327668, -0.30930856 # preview
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2361085,-1.2319995, 1.2294204, -1.210581, 0.5923838, -0.62981504
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float64)
        lbl = torch.from_numpy(lbl).float()
        labels['wc_gt'] = lbl

        curv_mx = 0.02297426 #curvature

        for label in labels:
            if label != 'wc_gt':
                

                lbl = labels[label]
                if label == 'warped_curvature_gt':
                    lbl[:,:,0]= (lbl[:,:,0]/curv_mx)


                lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
                lbl = np.array(lbl, dtype=np.float64)
                lbl = torch.from_numpy(lbl).float()
                labels[label] = lbl
                
        return labels

class Dataset_backward_mapping(Dataset):

    def __init__(self, data_dir, transform=True, img_size = 256):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        idx = sorted(os.listdir(self.data_dir))[idx]
        data_path = os.path.join(self.data_dir, idx)

        input = np.load(data_path + '/warped_WC.npz')['warped_WC']
        
        labels = {}
        labels['warped_bm'] = np.load(data_path + '/warped_BM.npz')['warped_BM']
        labels['warped_uv'] = np.load(data_path + '/warped_UV.npz')['warped_UV']
        labels['warped_angle'] = np.load(data_path + '/warped_angle.npz')['warped_angle']
        labels['warped_text_mask'] = np.load(data_path + '/warped_text_mask.npz')['warped_text_mask']
        labels['img'] = cv2.imread(data_path + '/warped_document.png')

        if self.transform:
            input, labels = self.tight_crop_all(input,labels)
            input, labels = self.transform_data(input, labels)

        return input, labels
    
    def tight_crop_all(self, fm, labels):
        
        msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
    
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        
        fm = fm[miny : maxy + 1, minx : maxx + 1, :]
        fm = cv2.resize(fm, self.img_size, interpolation=cv2.INTER_NEAREST)

        for label in labels:
            if label != 'warped_bm':
                im = labels[label]
                im = np.array(im, dtype=np.float64)
                #ic(label, im.shape)
                im = im[miny : maxy + 1, minx : maxx + 1, :]
                im = cv2.resize(im, self.img_size, interpolation=cv2.INTER_NEAREST)
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

                bm = bm*448
                bm[:,:,1]=bm[:,:,1]-t
                bm[:,:,0]=bm[:,:,0]-l
                bm=bm/np.array([float(448)-l-r, float(448)-t-b])
                bm=(bm-0.5)*2
                bm = cv2.resize(bm, self.img_size, interpolation=cv2.INTER_NEAREST)
                labels[label] = bm

        return fm, labels

    def transform_data(self, input, labels):
        lbl = input
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2361085,-1.2319995, 1.2294204, -1.210581, 0.5923838, -0.62981504
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
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


class Dataset_full_model(Dataset):

    def __init__(self, data_dir, transform=True, img_size = 256):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        idx = sorted(os.listdir(self.data_dir))[idx]
        data_path = os.path.join(self.data_dir, idx)

        
        input =  cv2.imread(data_path + '/warped_document.png')

        labels = {}
        labels['warped_bm'] = np.load(data_path + '/warped_BM.npz')['warped_BM']
        labels['warped_uv'] = np.load(data_path + '/warped_UV.npz')['warped_UV']
        labels['warped_angle'] = np.load(data_path + '/warped_angle.npz')['warped_angle']
        labels['warped_text_mask'] = np.load(data_path + '/warped_text_mask.npz')['warped_text_mask']
        labels['wc_gt'] = np.load(data_path + '/warped_WC.npz')['warped_WC'] #world coordinates ground truth
        labels['warped_curvature_gt'] = np.load(data_path + '/warped_curvature.npz')['warped_curvature']
        
        if self.transform:
            input, labels = self.tight_crop_all(input, labels)
            input, labels = self.transform_data(input, labels)

        return input, labels

    def tight_crop_all(self, img, labels):
        fm = labels['wc_gt']
        msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
    
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        
        fm = fm[miny : maxy + 1, minx : maxx + 1, :]
        fm = cv2.resize(fm, self.img_size, interpolation=cv2.INTER_NEAREST)
        labels['wc_gt'] = fm
        img = img[miny : maxy + 1, minx : maxx + 1, :]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)

        for label in labels:
            if label != 'wc_gt':
                if label != 'warped_bm':
                    im = labels[label]
                    im = np.array(im, dtype=np.float64)
                    #ic(label, im.shape)
                    im = im[miny : maxy + 1, minx : maxx + 1, :]
                    im = cv2.resize(im, self.img_size, interpolation=cv2.INTER_NEAREST)
                    if label == 'warped_text_mask' or label == 'warped_curvature_gt':
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

                    bm = bm*448
                    bm[:,:,1]=bm[:,:,1]-t
                    bm[:,:,0]=bm[:,:,0]-l
                    bm=bm/np.array([float(448)-l-r, float(448)-t-b])
                    bm=(bm-0.5)*2
                    bm = cv2.resize(bm, self.img_size, interpolation=cv2.INTER_NEAREST)
                    labels[label] = bm

        return img, labels

    def transform_data(self, input, labels):

        # convert image to numpy array
        img = asarray(input)
        #img = m.imresize(img, self.img_size) # uint8 with RGB mode
        if img.shape[-1] == 4:
            img=img[:,:,:3]   # Discard the alpha channel  
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        input = torch.from_numpy(img).float()

        
        lbl = labels['wc_gt']
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2361085,-1.2319995, 1.2294204, -1.210581, 0.5923838, -0.62981504
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float64)
        lbl = torch.from_numpy(lbl).float()
        labels['wc_gt'] = lbl

        curv_mx = 0.02297426 #curvature

        for label in labels:
            if label != 'wc_gt':

                if label != 'img':
                    lbl = labels[label]
                    if label == 'warped_curvature_gt':
                        lbl[:,:,0]= (lbl[:,:,0]/curv_mx)

                    lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
                    lbl = np.array(lbl, dtype=np.float64)
                    lbl = torch.from_numpy(lbl).float()
                    labels[label] = lbl
                else:
                    img = labels[label]
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).float()
                    #img = img.unsqueeze(0)#.transpose(2,3).transpose
                    img = img / 255
                    labels[label] = img

        return input, labels

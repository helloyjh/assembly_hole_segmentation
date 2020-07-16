import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
class ImageDataLoader(Dataset):

    def __init__(self, list, path, transform=None, transform2=None):
        self.transform = transform
        self.transform2 = transform2
        self.list = list
        self.path = path
        self.depth_min = 1.0
        self.depth_max = 2.3

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        file_name = self.list[index]
        img_name = self.path+'/rgb/'+file_name
        
        #mask_name = self.path+'/'+file_name
        #part_mask_name = self.path+'/'+file_name

        depth_file_name = file_name.split('.png')[0]+'.npy'
        depth_name = self.path+'/depth_value/'+depth_file_name
        #print('imgname:',img_name)
        #print('dname:',depth_name)
        
        image = cv2.imread(img_name)
        #print(image.shape)
        image = self.transform(Image.fromarray(image))
        #print(image.shape)
        #exit()
        #mask = cv2.imread(mask_name)
        #mask = mask[:, :, 0]
        #mask = self.transform2(Image.fromarray(mask))
        
        #part_mask = cv2.imread(part_mask_name)
        #part_mask = part_mask[:,:,2]
        #vals = np.unique(part_mask)
        #cnd = part_mask[:,:]>0
        #part_mask[cnd]=255
        #part_mask = self.transform2(Image.fromarray(part_mask))
        
        depth = np.load(depth_name).astype('int16')
        depth = cv2.resize(depth, dsize=(640, 400), interpolation=cv2.INTER_CUBIC) 
        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth-self.depth_min)/(self.depth_max- self.depth_min)
        #depth = np.transpose(depth, [1,0])
        depth = torch.from_numpy(depth).unsqueeze(dim=0)
        
        return image, depth, file_name

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import cv2
import numpy as np
from PIL import Image
import collections
from torch import optim
import loader_siminf as loader
from Model2 import Model
import glob
import matplotlib as mpl
import math
from unet import UNet
import pickle

mpl.use('Agg')

arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default="0")
arg.add_argument('--bs', default='60')
arg.add_argument('--lr', default='0.0001')
arg.add_argument('--wd', default='0')
arg.add_argument('--st', default='0')  #start epoch
arg.add_argument('--ep', default='50')  #train epochs
arg = arg.parse_args()

epochs=int(arg.ep)
start_epoch = int(arg.st)
embedding_size=256
batch_size=int(arg.bs)
gpu_id=arg.gpu
seed=0

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
np.random.seed(seed)

transform = transforms.Compose([
                         transforms.Resize((320, 200)),
                         #transforms.RandomAffine(0,translate=(0,0.15)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                     ])

transform2 = transforms.Compose([
                        transforms.Resize((320, 200)),
                        transforms.ToTensor()
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #          std=[0.229, 0.224, 0.225])
                        ])
with open('list.pickle', 'rb') as f:
    dataset_lists = pickle.load(f)

test_dir = dataset_lists[2]
test_dir1 = []
for i in range(30):
    test_dir1.append(test_dir[i])
test_dir = test_dir1

testset = loader.ImageDataLoader(test_dir, '../SIM_dataset_v10/', transform=transform, transform2=transform2)
test_loader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

def Transpose(x,y,m):
    k = 1/(1+(m*m))
    X = (1-(m*m))*x + 2*m*y
    Y = 2*m*x + y*((m*m)-1)
    X = k*X
    Y = k*Y
    return X, Y

def main():
    model = nn.DataParallel(UNet(n_channels=4, n_classes=2, bilinear=True))
    model.cuda()
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    count = 0
    for batch_idx, data in enumerate(test_loader):
        img = data[0].cuda()
        mask = data[1].cuda()
        part_mask = data[2].cuda()
        depth = data[3].cuda().float()
        img_name = data[4][0]
        
        input_data = torch.cat([img, depth], dim=1)
        #print(input_data.shape)
        #img = img * part_mask
                
        cls = model(input_data).squeeze(dim=0)
        #print(cls.shape)
        
        cls_b = cls.cpu().detach().numpy()
        #img = img.squeeze(dim=0).cpu().detach().numpy()
        #cls_b = np.transpose(cls_b, (1,2,0))
       
        #img = np.transpose(img, (2,1,0))
        nameload = '../SIM_dataset_v10/rgb/'+test_dir[batch_idx]
        real_img = cv2.imread(nameload)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = real_img.shape
        

        #print('W:',img_w)
        #print('H:',img_h)
        #real_img = cv2.warpAffine(real_img, cv2.getRotationMatrix2D((int(img_shape[1] / 2), int(img_shape[0]/ 2)), 180, 1),(img_shape[1], img_shape[0]))

        #nf_array = np.zeros(320,200)
        # print(cls_b.shape)
        # exit()
        #cv2.imwrite('report/' + img_name[0]+'.png', img)
        for w in range(320):
            for h in range(200):
                #print(cls_b[h][w][0])
                #print(cls_b[h][w][1])
                if cls_b[0][w][h] < cls_b[1][w][h]:
                    print('yes')
                    #x,y = Transpose(w,h,0.625)
                    cv2.circle(real_img, (int(w*img_w/320),int(h*img_h/200)), 13, [250, 0, 0], -1)                  
                    #cv2.circle(real_img, (int(w*w_img/320), int(h)), 13, [150,0,0], -1)
        #cv2.circle(real_img, (0,0), 15, [0,0,150], -1)
        #cv2.circle(real_img, (320,200), 15, [0,0,150],-1)
        #cv2.circle(real_img, (640,400), 15, [0,0,150], -1)
        #cv2.circle(real_img, (960, 600), 15, [0,0,150], -1)
        #cv2.circle(real_img, (1280, 800), 15, [0,0,150], -1)
        #cv2.circle(real_img, (1600, 1000), 15, [0,0,150], -1)
        #cv2.circle(real_img, (1920, 1200), 15, [0,0,150], -1)
                    #real_img[3*h][3*w][2]=255
                # else :
                #     img[3*h][3*w][0] = 0
                #     img[3 * h][3 * w][1] = 0
                #     img[3 * h][3 * w][2] = 0
        st_name = img_name.split('/')
        cv2.imwrite('report/'+st_name[len(st_name)-1], real_img)

if __name__ == '__main__':
    main()

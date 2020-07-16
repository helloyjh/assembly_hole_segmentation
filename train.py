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
import numpy as np
from PIL import Image
import collections
from torch import optim
import loader
from Model2 import Model
import glob
import matplotlib as mpl
from unet import UNet
import cv2
import pickle
from termcolor import colored, cprint
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

mpl.use('Agg')

arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default="0")
arg.add_argument('--bs', default='12')
arg.add_argument('--lr', default='0.0005')
arg.add_argument('--wd', default='0.00001')
arg.add_argument('--st', default='0')  #start epoch
arg.add_argument('--ep', default='100')  #train epochs
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
                         transforms.Resize((400, 640)),
                         #transforms.RandomAffine(0,translate=(0,0.15)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                     ])

transform2 = transforms.Compose([
                        transforms.Resize((400, 640)),
                        transforms.ToTensor()
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #          std=[0.229, 0.224, 0.225])
                        ])

with open('list.pickle', 'rb') as f:
    dataset_lists = pickle.load(f)
train_dir = dataset_lists[0]
val_dir = dataset_lists[1]
print('train:',len(train_dir),'val:',len(val_dir))
train_dir1 = []
val_dir1=[]

for i in range(len(train_dir)):
    if i%5==0:
        train_dir1.append(train_dir[i])
for i in range(len(val_dir)):
    if i%5==0:
        val_dir1.append(val_dir[i])
train_dir = train_dir1
val_dir= val_dir1

print(colored('CUT:', 'cyan'), 'train:',len(train_dir),'val:',len(val_dir))
trainset = loader.ImageDataLoader(train_dir, transform=transform, transform2=transform2)
valset = loader.ImageDataLoader(val_dir, transform=transform, transform2=transform2)
train_loader=torch.utils.data.DataLoader(trainset, batch_size=4,  shuffle=True, drop_last=True)#, num_workers=1)
val_loader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

def main():
    model = nn.DataParallel(DeepLab(num_classes=2, backbone='xception'))
    model.cuda()
    #model.load_state_dict(torch.load('best_model_2.pt'))
    optimizer = optim.Adagrad(model.parameters(),lr=float(arg.lr))

    train_losses = []
    avg_train_losses = []
    valid_losses = []
    avg_valid_losses = []
    n_epochs = epochs
    print('train epochs : {}'.format(n_epochs))
    count = 0
    best_loss = float(1000.0)
    epoch_store = []
    train_loss_st = []
    val_loss_st = []
    stop_cnt = 0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        loss_add_time = 0
        for batch_idx, data in enumerate(train_loader):
            #print('22222222222222')
            img = data[0].cuda()
            mask = data[1].cuda().long().squeeze(dim=1)
            part_mask = data[2].cuda()
            depth = data[3].cuda().float()
            
            input_data = torch.cat([img, depth], dim=1)
        
            #part_mask = data[2].cuda()

            #img = img*part_mask
            
            cls = model(input_data)
            
            weights = torch.FloatTensor([1,6153]).cuda()
            criterion = nn.CrossEntropyLoss(weight=weights)

            loss = criterion(cls, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            loss_add_time += 1
            loading = 100*(batch_idx/len(train_loader))
            print('loading... : {}%\r'.format(round(loading,1)),end='')
        
        print('---------- epoch    {}   ----------'.format(epoch))
        print('training loss: {:.8f}'.format(train_loss/loss_add_time))

        train_loss_st.append(train_loss/loss_add_time)

        model.eval()
        val_loss=0
        for batch_idx, data in enumerate(val_loader):
            img = data[0].cuda()
            mask = data[1].cuda().long().squeeze(dim=1)
            part_mask = data[2].cuda()
            depth = data[3].cuda().float()
                 
            input_data = torch.cat([img, depth], dim=1)
            #img= img * part_mask

            cls = model(input_data)

            weights = torch.FloatTensor([1,6153]).cuda()
            criterion = nn.CrossEntropyLoss(weight=weights)
    
            loss = criterion(cls, mask)

            val_loss = val_loss + loss.item()

            loading = 100*(batch_idx/len(val_loader))
            print('val loading... : {:.1f}%\r'.format(loading), end='')
            
        epoch_store.append(epoch)
        val_loss_st.append(val_loss/len(val_loader))
        if val_loss/len(val_loader) < best_loss:
            best_loss = val_loss/len(val_loader)
            torch.save(model.state_dict(), 'best_model.pt')
            stop_cnt = 0
        else : stop_cnt +=1

        if stop_cnt ==15:
            break

        print('val_loss: {:.8f}'.format(val_loss/len(val_loader)))

    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('train and val loss')
    plt.plot(epoch_store, train_loss_st, 'b')
    plt.plot(epoch_store, val_loss_st, 'r')
    plt.savefig('report/loss_st{}_lr{}_bs{}_wd{}.png'.format(arg.st,arg.lr, arg.bs, arg.wd))

if __name__ == '__main__':
    main()

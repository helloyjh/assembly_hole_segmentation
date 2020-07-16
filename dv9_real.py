import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from termcolor import colored
import numpy as np
scale_factor = 3
root = 'inference_data_zivid_white_2/depth_value'
new_root = 'REAL_dv9_2/depth_value'
#os.mkdir(new_root.split('/')[0])
os.mkdir(new_root)
files = os.listdir(root)
print(colored('from {} files, generating {} new files '.format( len(files), len(files)*scale_factor*scale_factor), 'cyan'))
for k in range(len(files)):
    print('{}/{}'.format(k+1, len(files)))
    #file = cv2.imread(root+'/'+files[k])
    file = np.load(root+'/'+files[k])
    width_orginal = file.shape[1]
    height_original = file.shape[0]

    width = int(width_orginal / scale_factor)
    height = int(height_original / scale_factor)
    #print(width, height)
    pic_list = []
    pic1 = file[:height,:width]
    pic2 = file[:height,width:2*width]
    pic3 = file[:height,2*width:3*width]
    pic4 = file[height:2*height,:width]
    pic5 = file[height:2*height,width:2*width]
    pic6 = file[height:2*height,2*width:3*width]
    pic7 = file[2*height:3*height,:width]
    pic8 = file[2*height:3*height,width:2*width]
    pic9 = file[2*height:3*height,2*width:3*width]
    pic_list.append(pic1)
    pic_list.append(pic2)
    pic_list.append(pic3)
    pic_list.append(pic4)
    pic_list.append(pic5)
    pic_list.append(pic6)
    pic_list.append(pic7)
    pic_list.append(pic8)
    pic_list.append(pic9)

    for i in range(9):
        np.save(new_root+'/{}of9'.format(i+1)+files[k],pic_list[i])
        #cv2.imwrite(new_root+'/{}of9'.format(i+1)+files[k], pic_list[i])

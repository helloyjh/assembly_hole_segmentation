import torch
from PIL import Image
import numpy
import cv2
# x = torch.FloatTensor([[[1,0,0,10,20],[1,1,1,20,50]],[[1,0,0,10,20],[1,1,1,20,50]]])
# y = torch.FloatTensor([[[1,1,1,1,1],[0,0,0,0,0]]])
# print(x.shape)
# print(y.shape)
# print(x*y)
i = cv2.imread('train/part_mask/3.png_r0.png.png')
i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
print(i.shape)
img = Image.fromarray(i)
pix = numpy.array(img.getdata()).reshape(img.size[0], img.size[1], 1)
print(pix.shape)
print()
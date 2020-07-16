import os
import cv2
import numpy as np

hole_masks = os.listdir('../SIM_dataset_v10/hole_mask')
print('len:',len(hole_masks))
tots = 0
ones = 0
for p in range(len(hole_masks)):
    #print('{}/{}'.format(p, len(hole_masks)))
    q = cv2.imread('../SIM_dataset_v10/hole_mask/{}'.format(hole_masks[p]))
    q = q[:,:,0]
    cnd = q[:,:]>0
    pos = np.sum(cnd)
    ones = ones + pos
    tot = cnd.shape[0]*cnd.shape[1]
    tots = tots + tot

print('1:',ones, '0:',tots)
ratio = int((tots-ones)/ones)
print(ratio)

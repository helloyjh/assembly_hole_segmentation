import os
import pickle
import cv2
import numpy as np
import random
import shutil

list_a = []
fff = os.listdir('gened_rgb')
for i in range(420):
    shutil.move('gened_rgb/'+fff[i],'train/img/'+fff[i])
    shutil.move('gened_mask/' + fff[i], 'train/mask/' + fff[i])
    shutil.move('gened_part_mask/' + fff[i], 'train/part_mask/' + fff[i])
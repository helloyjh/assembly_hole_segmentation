import os
import pickle
import cv2
import numpy as np
import random
import shutil

list_a = []
fff = os.listdir('gened_rgb')
for i in range(len(fff)):
    shutil.move('gened_rgb/'+fff[i],'val/img/'+fff[i])
    shutil.move('gened_mask/' + fff[i], 'val/mask/' + fff[i])
    shutil.move('gened_part_mask/' + fff[i], 'val/part_mask/' + fff[i])
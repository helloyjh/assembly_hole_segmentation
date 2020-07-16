import os
import pickle
import cv2
import numpy as np
import random
import shutil
from termcolor import colored, cprint

def dataset_list(train_rate=0.7, val_rate=0.25):
    random.seed(0)
    fff = os.listdir('../SIM_v10_dv9/rgb')
    train_len = int(train_rate*(len(fff)))
    val_len = int(val_rate*(len(fff)))

    list_train = []
    ran_num = random.randint(0,len(fff)-1)
    for i in range(train_len):
        while fff[ran_num] in list_train :
            ran_num = random.randint(0,len(fff)-1)
        list_train.append(fff[ran_num])
    list_train.sort()


    for i in range(len(list_train)):
        fff.remove(list_train[i])


    list_val = []
    ran_num = random.randint(0,len(fff)-1)
    for i in range(val_len):
        while fff[ran_num] in list_val :
            ran_num = random.randint(0,len(fff)-1)
        list_val.append(fff[ran_num])
    list_val.sort()

    for i in range(len(list_val)):
        fff.remove(list_val[i])

    print(colored('dataset lists are called : train {} val {} test {}'.format(len(list_train), len(list_val), len(fff)), 'blue'))
    return [list_train,list_val,fff]

a = dataset_list(0.7, 0.25)

with open('list.pickle', 'wb') as f:
    pickle.dump(a, f)

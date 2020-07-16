import cv2
import os

files = os.listdir('inference/img')

for i in range(len(files)):
    img = cv2.imread('inference/img/{}'.format(files[i]))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, c = img.shape

    for k in range(3):
        img_rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((int(width/2), int(height/2)),15*k,1), (width, height))
        cv2.imwrite('inference/img/{}_r{}.png'.format(files[i], k), img_rot)

    mask = cv2.imread('inference/mask/{}'.format(files[i]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width = mask.shape
    for h in range(height):
        for w in range(width):
            if mask[h][w]>0:
                mask[h][w]=255

    for k in range(3):
        img_rot = cv2.warpAffine(mask, cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), 15 * k, 1),(width, height))
        cv2.imwrite('inference/mask/{}_r{}.png'.format(files[i], k), img_rot)

    pmask = cv2.imread('inference/part_mask/{}'.format(files[i]))
    pmask = cv2.cvtColor(pmask, cv2.COLOR_BGR2GRAY)
    height, width = pmask.shape
    for h in range(height):
        for w in range(width):
            if pmask[h][w]>0:
                pmask[h][w]=255
    for k in range(3):
        img_rot = cv2.warpAffine(pmask, cv2.getRotationMatrix2D((int(width/2), int(height/2)),15*k,1), (width, height))
        cv2.imwrite('inference/part_mask/{}_r{}.png'.format(files[i], k), img_rot)
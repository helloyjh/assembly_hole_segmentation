import cv2
import os

files = os.listdir('inference/img')

for i in range(len(files)):
    img = cv2.imread('inference/img/{}'.format(files[i]))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, c= img.shape

    img_rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((int(width/2), int(height/2)),180,1), (width, height))
    img_hflip = cv2.flip(img, 1)
    img_vflip = cv2.flip(img, 0)
    #cv2.imwrite('inference/img/{}.png'.format(files[i]), img)
    cv2.imwrite('inference/img/{}_rot.png'.format(files[i]), img_rot)
    cv2.imwrite('inference/img/{}_hflip.png'.format(files[i]), img_hflip)
    cv2.imwrite('inference/img/{}_vflip.png'.format(files[i]), img_vflip)

    mask = cv2.imread('inference/mask/{}'.format(files[i]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width  = mask.shape

    mask_rot = cv2.warpAffine(mask, cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), 180, 1), (width, height))
    mask_hflip = cv2.flip(mask, 1)
    mask_vflip = cv2.flip(mask, 0)
    #cv2.imwrite('inference/mask/{}.png'.format(files[i]), mask)
    cv2.imwrite('inference/mask/{}_rot.png'.format(files[i]), mask_rot)
    cv2.imwrite('inference/mask/{}_hflip.png'.format(files[i]), mask_hflip)
    cv2.imwrite('inference/mask/{}_vflip.png'.format(files[i]), mask_vflip)

    pmask = cv2.imread('inference/part_mask/{}'.format(files[i]))
    pmask = cv2.cvtColor(pmask, cv2.COLOR_BGR2GRAY)
    height, width = pmask.shape

    pmask_rot = cv2.warpAffine(pmask, cv2.getRotationMatrix2D((int(width / 2), int(height / 2)), 180, 1), (width, height))
    pmask_hflip = cv2.flip(pmask, 1)
    pmask_vflip = cv2.flip(pmask, 0)
    #cv2.imwrite('inference/part_mask/{}.png'.format(files[i]), pmask)
    cv2.imwrite('inference/part_mask/{}_rot.png'.format(files[i]), pmask_rot)
    cv2.imwrite('inference/part_mask/{}_hflip.png'.format(files[i]), pmask_hflip)
    cv2.imwrite('inference/part_mask/{}_vflip.png'.format(files[i]), pmask_vflip)
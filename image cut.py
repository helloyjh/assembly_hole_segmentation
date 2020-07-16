import cv2
import os

files = os.listdir('rgb_inf')
for i in range(len(files)):
    img = cv2.imread('rgb_inf/{}'.format(files[i]))
    #2048/1536
    crop_img = img[650:1536, 500:1700]
    # cv2.imshow("crppoed", crop_img)
    # cv2.waitKey(0)
    # exit()
    cv2.imwrite('inference/img/{}'.format(files[i]), crop_img)

    mask = cv2.imread('mask_inf/{}'.format(files[i]))
    crop_mask = mask[650:1536, 500:1700]
    cv2.imwrite('inference/mask/{}'.format(files[i]), crop_mask)

    part_mask = cv2.imread('part_mask_inf/{}'.format(files[i]))
    crop_mask = part_mask[650:1536, 500:1700]
    cv2.imwrite('inference/part_mask/{}'.format(files[i]), crop_mask)


import os
import cv2
import numpy as np

mask_dir = r'E:\DATASET\pku-autonomous-driving\test_masks'
test_dir =r'E:\DATASET\pku-autonomous-driving\test_images'
output_dir = r'E:\DATASET\pku-autonomous-driving\wudi_data\ignore_mask_color_shift'

mask_list = os.listdir(mask_dir)
bgr_threshold = 10

for img_name in mask_list:
    img_mask = cv2.imread(os.path.join(mask_dir, img_name))
    b_max, g_max, r_max = img_mask[:, :, 0].max(), img_mask[:, :, 1].max(), img_mask[:, :, 2].max()
    if (img_mask[:, :, 0].max() != 255) or (img_mask[:, :, 1].max() != 255) or (img_mask[:, :, 2].max() != 255):
        # print("Max altered in image %s" % img_name)
        print("Max B: %d, Max G: %d, Max R: %d" % (img_mask[:, :, 0].max(), img_mask[:, :, 1].max(), img_mask[:, :, 2].max()))
        if np.abs(b_max - g_max) > bgr_threshold or np.abs(b_max - r_max) > bgr_threshold or np.abs(r_max - g_max) > bgr_threshold:
            print("Color alternation for %s" % img_name)
            #color_max = max(max(b_max, g_max), r_max)
            color_max = 255

            multiple_ratio = color_max/b_max, color_max/g_max, color_max/r_max

            test_img = cv2.imread(os.path.join(test_dir, img_name))
            img_mask_return = test_img.copy().astype(float)
            img_mask_return[:, :, 0] *= multiple_ratio[0]
            img_mask_return[:, :, 1] *= multiple_ratio[1]
            img_mask_return[:, :, 2] *= multiple_ratio[2]
            img_mask_return = img_mask_return.astype(np.uint8)

            cv2.imwrite(os.path.join(output_dir, img_name), test_img)
            cv2.imwrite(os.path.join(output_dir, img_name[:-4]+'_return.jpg'), img_mask_return)

    if (img_mask[:, :, 0].min() != 0) or (img_mask[:, :, 1].min() != 0) or (img_mask[:, :, 2].min() != 0):
        # print("Min altered in image %s" % img_name)
        # print("Min B: %d, Min G: %d, Max R: %d" % (img_mask[:, :, 0].min(), img_mask[:, :, 1].min(), img_mask[:, :, 2].min()))
        pass

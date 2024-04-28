import os
import cv2
import numpy as np

def extract_labels(img, labels_save_path, img_rgb):
    exclude_tags = [0, 1, 23]
    target_tags = np.unique(img)
    target_tags = [tag for tag in target_tags if tag not in exclude_tags]
    target_tags.sort()
    img_x, img_y = img.shape

    with open(labels_save_path, 'w') as f:
        for tag in target_tags:
            target = np.zeros_like(img, dtype=np.uint8)
            target[img == tag] = 1
            contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # res = cv2.drawContours(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), contours, -1, (0, 255, 0), 1)
            # cv2.imwrite('/raid/home-robot/gyzp/test/' + labels_save_path.split('/')[-1].split('.')[0] + '-' + str(tag) + '.png', res)
            
            for contour in contours:
                contour = contour.flatten().astype(np.float32)
                contour[0::2] /= img_x  # scale x
                contour[1::2] /= img_y  # scale y
                contour = contour.tolist()
                contour = ['{:.6f}'.format(i) for i in contour]
                line = str(tag-2) + " " + ' '.join(contour)
                f.write(line + '\n')

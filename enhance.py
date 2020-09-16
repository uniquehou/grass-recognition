import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


dir = 'src'
label_dir = 'label'
to_dir = 'EnSrc'


def getGreen(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,1] += np.minimum( np.full(img[..., 1].shape, 100, dtype=np.uint8), np.array( (255-img[..., 1]) *0.5, dtype=np.uint8))
    img[:,:,2] -= np.array( img[..., 2] *0.3, dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def toEnhance(dir, to_dir):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    names = os.listdir(dir)
    for name in names:
        img = cv2.imread(os.path.join(dir, name))
        green = getGreen(img)
        cv2.imwrite(os.path.join(to_dir, name), green)

def show():
    img_names = os.listdir(dir)
    show_count = 3
    for i in np.random.choice(len(img_names), show_count, replace=False):
        img = cv2.imread(os.path.join(dir, img_names[i]))
        label = cv2.resize( cv2.imread(os.path.join(label_dir, img_names[i]), flags=0), img.shape[:-1])

        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        green_img = getGreen(img)
        plt.imshow(cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 3)
        plt.imshow(label)
        plt.show()

if __name__ == '__main__':
    toEnhance(dir, to_dir)
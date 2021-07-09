
import os
import shutil

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('tkagg')

IMAGES_PATH = "/media/andrey/big/downloads/train/"
MY_LABELS_PATH = "/media/andrey/big/downloads/my_labels"
SAVE_DIR = "/media/andrey/big/preprocess/HANDS_MY"

IMAGES_SAVE_DIR = os.path.join(SAVE_DIR, "images")
LABELS_SAVE_DIR = os.path.join(SAVE_DIR, "labels")
if not os.path.exists(IMAGES_SAVE_DIR): os.makedirs(IMAGES_SAVE_DIR)
if not os.path.exists(LABELS_SAVE_DIR): os.makedirs(LABELS_SAVE_DIR)

def get_rec(x_center, y_center, w, h, image):
    x = (x_center - w / 2.) * image.shape[1]
    y = (y_center - h / 2.) * image.shape[0]
    w = w * image.shape[1]
    h = h * image.shape[0]
    return patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

for folder_name in os.listdir(IMAGES_PATH):
    folder_path = os.path.join(IMAGES_PATH, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_id = image_name.split('.')[0]
            label_path = os.path.join(MY_LABELS_PATH, image_id + ".txt")
            if os.path.exists(label_path):
                df = pd.read_csv(label_path, delimiter=' ', names=['class', 'x_center', 'y_center', 'width', 'height'])
                df['class'] = 0
                shutil.copy(image_path, os.path.join(IMAGES_SAVE_DIR, image_id + ".jpg"))
                df.to_csv(os.path.join(LABELS_SAVE_DIR, image_id + ".txt"), sep=' ', header=False, index=False)

                # fig, ax = plt.subplots()
                # image = cv2.imread(image_path)
                # ax.imshow(image)
                # for _, row in df.iterrows():
                #     ax.add_patch(get_rec(row.x_center, row.y_center, row.width, row.height, image))
                # plt.show()


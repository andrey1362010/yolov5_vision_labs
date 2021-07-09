import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pandas as pd
import json
import shutil

from tqdm import tqdm

IMAGES_PATH = "/media/andrey/big/downloads/100D/raw/"
LABELS_TRAIN_PATH = "/media/andrey/big/downloads/100D/raw/file/trainval.json"
LABELS_TEST_PATH = "/media/andrey/big/downloads/100D/raw/file/test.json"
SAVE_PATH = "/media/andrey/big/preprocess/HANDS"

def get_rec(box, image):
    x = image.shape[1] * box[0]
    y = image.shape[0] * box[1]
    w = image.shape[1] * (box[2] - box[0])
    h = image.shape[0] * (box[3] - box[1])
    return patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

def convert_box(box):
    center_x = (box[0] + box[2]) / 2.
    center_y = (box[1] + box[3]) / 2.
    w = (box[2] - box[0])
    h = (box[3] - box[1])
    assert w > 0 and h > 0 and center_x > 0 and center_y > 0
    return center_x, center_y, w, h

def process(labels_path, images_path, save_path):
    save_path_images = os.path.join(save_path, "images")
    save_path_labels = os.path.join(save_path, "labels")
    if not os.path.exists(save_path_images):
        os.makedirs(save_path_images)
    if not os.path.exists(save_path_labels):
        os.makedirs(save_path_labels)

    with open(labels_path) as json_file:
        data = json.load(json_file)

    for i, row in tqdm(enumerate(data)):
        boxes = {'class': [], 'x_center': [], 'y_center': [], 'width': [], 'height': []}
        for obj in data[row]:
            box = (obj['x1'], obj['y1'], obj['x2'], obj['y2'])
            box = convert_box(box)
            boxes['class'].append(0)
            boxes['x_center'].append(box[0])
            boxes['y_center'].append(box[1])
            boxes['width'].append(box[2])
            boxes['height'].append(box[3])
        image_path = os.path.join(images_path, row)
        image_save_path = os.path.join(save_path_images, str(i) + ".jpg")
        labels_save_path = os.path.join(save_path_labels, str(i) + ".txt")
        shutil.copy(image_path, image_save_path)
        df = pd.DataFrame.from_dict(boxes)
        df.to_csv(labels_save_path, sep=' ', header=False, index=False)

process(LABELS_TRAIN_PATH, IMAGES_PATH, os.path.join(SAVE_PATH, "train"))
process(LABELS_TEST_PATH, IMAGES_PATH, os.path.join(SAVE_PATH, "test"))
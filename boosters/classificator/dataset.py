import os
import random

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

LABEL_INDEX_DICT = {
    'no_gesture': 0,
    'like': 1,
    'dislike': 2,
    'victory': 3,
    'ok': 4,
    'stop': 5,
    'mute': 6,
}
INDEX_LABEL_DICT = {v: k for k, v in LABEL_INDEX_DICT.items()}

def load_other_labels(path):
    labels = []
    for folder_name in tqdm(os.listdir(path)):
        if len(os.listdir(os.path.join(path, folder_name))) > 0:
            image_folder = folder_name.split("_")[0]
            image_name = folder_name.split("_")[1]
            class_name = 'no_gesture'
            labels.append((path, image_folder, image_name, class_name))
    print("OTHER LEN:", len(labels))
    return labels

def load_labels(images_path, path):
    df = pd.read_csv(path)
    labels = []
    for _, row in tqdm(df.iterrows()):
        frame_path = row['frame_path']
        image_folder = frame_path.split('/')[-2]
        image_name = frame_path.split('/')[-1].split(".")[0]
        class_name = row['class_name']
        image_folder_path = os.path.join(images_path, image_folder + "_" + image_name)
        if os.path.exists(image_folder_path):
            labels.append((images_path, image_folder, image_name, class_name))
    return labels

def test_train_split(lables):
    users_dict = {}
    for label in lables:
        path, image_folder, image_name, class_name = label
        if image_folder not in users_dict: users_dict[image_folder] = []
        users_dict[image_folder].append(label)
    SPLIT_SIZE = 200
    train_users = list(users_dict.keys())[:-SPLIT_SIZE]
    test_users = list(users_dict.keys())[-SPLIT_SIZE:]
    labels_train = []
    for user in train_users:
        labels_train.extend(users_dict[user])
    labels_test = []
    for user in test_users:
        labels_test.extend(users_dict[user])
    return labels_train, labels_test

MAX_IMAGES = 6
IMAGE_SIZE = 116

class HandDataset(torch.utils.data.Dataset):

    def __init__(self, labels, test=False):
        self.labels = labels
        self.test = test
        self.seq = iaa.Sequential([
            iaa.GammaContrast((0.4, 3.0)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
            iaa.Grayscale(alpha=(0.0, 0.5)),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),

            iaa.Fliplr(0.5),
            iaa.GaussianBlur(sigma=(0, 4.0)),
            iaa.CropAndPad(
                px=(-15, 15), #(-24, 24)
                sample_independently=False
            ),
            iaa.Rotate((-20, 20)),
        ])

    def __getitem__(self, index):
        path, image_folder, image_name, class_name = self.labels[index]
        images_dir_name = image_folder + "_" + image_name
        images_dir_path = os.path.join(path, images_dir_name)
        images_names = os.listdir(images_dir_path)
        random.shuffle(images_names)

        images = []
        for name in images_names:
            image = cv2.imread(os.path.join(images_dir_path, name))
            # if random.random() > 0.5 and not self.test:
            #     resize_size = random.randint(50, 116)
            #     image = cv2.resize(image, (resize_size, resize_size))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(image)

        if not self.test:
            images = self.seq(images=images)
            # plt.imshow(images[0])
            # plt.show()

        data = []
        for i in range(min(MAX_IMAGES, len(images_names))):
            image = images[i]
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float64) / 255.
            data.append(image)

        class_index = LABEL_INDEX_DICT[class_name]
        return np.array(data), len(data), class_index

    def __len__(self):
        return len(self.labels)

def collacate(batch):
    images = np.concatenate([i for i, l, c in batch], axis=0)
    return images, [l for i, l, c in batch], np.array([c for i, l, c in batch])

# LABELS_PATH = "/media/andrey/big/downloads/train.csv"
# IMAGES_PATH = "/media/andrey/big/preprocess/HANDS_CROPPED"
# labels = load_labels(LABELS_PATH)
# dataset = HandDataset(labels, IMAGES_PATH)
# d, c = next(iter(dataset))
# print(d.shape, c)
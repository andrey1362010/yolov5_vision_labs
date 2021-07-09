import os
import random

import pandas as pd

from boosters.classificator.dataset import load_labels, LABEL_INDEX_DICT

LABELS_PATH = "/media/andrey/big/downloads/train.csv"
REAL_IMAGES_PATH = "/media/andrey/big/downloads/train"
FAKE_IMAGES_PATH = "data/"

labels = load_labels(REAL_IMAGES_PATH, LABELS_PATH)
random.shuffle(labels)
paths_list = []
class_list = []
for path, image_folder, image_name, class_name in labels:
    if len(paths_list) > 10_000: break
    fake_path = os.path.join(path, image_folder, image_name + ".jpg")
    paths_list.append(fake_path)
    class_list.append(LABEL_INDEX_DICT[class_name])


result_df = pd.DataFrame({'frame_path': paths_list, "class": class_list})

result_df.to_csv(os.path.join(REAL_IMAGES_PATH, "test_local3.csv"), index=False)
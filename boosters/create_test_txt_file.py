import os
import pandas as pd

LABELS_PATH = "/media/andrey/big/downloads/train.csv"
REAL_IMAGES_PATH = "/media/andrey/big/downloads/train"
FAKE_IMAGES_PATH = "data/"

paths_list = []
class_list = []
for folder in os.listdir(REAL_IMAGES_PATH):
    folder_path = os.path.join(REAL_IMAGES_PATH, folder)
    if len(paths_list) > 3_000: break
    for name in os.listdir(folder_path):
        #fake_path = os.path.join(FAKE_IMAGES_PATH, folder, name)
        fake_path = os.path.join(REAL_IMAGES_PATH, folder, name)
        paths_list.append(fake_path)

result_df = pd.DataFrame({'frame_path': paths_list})

result_df.to_csv(os.path.join(REAL_IMAGES_PATH, "test_local2.csv"), index=False)
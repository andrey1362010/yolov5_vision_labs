import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from boosters.classificator.dataset import LABEL_INDEX_DICT, INDEX_LABEL_DICT

INPUT_PATH = "/media/andrey/big/downloads/train/test_local2.csv"
ANSWERS_PATH = "/home/andrey/IdeaProjects/yolov5/answers.csv"
SAVE_PATH = "/media/andrey/big/preprocess/HANDS_HARD"

input_df = pd.read_csv(INPUT_PATH)
answer_df = pd.read_csv(ANSWERS_PATH)

df = pd.merge(answer_df, input_df, left_on='frame_path', right_on='frame_path')

def roc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    index_roc = np.argmax(np.array(fpr) > 0.002)
    score = tpr[index_roc]
    return score

dir_index = 0
count = 0
for i, row in df.iterrows():
    dir_path = os.path.join(SAVE_PATH, str(dir_index))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    probs = np.array([row[INDEX_LABEL_DICT[i]] for i in range(len(INDEX_LABEL_DICT))])
    index = probs.argmax()
    if index != row['class']:
        shutil.copy(row.frame_path, os.path.join(dir_path, row.frame_path.split('/')[-1]))
        count += 1
        if count >= 500:
            count = 0
            dir_index += 1


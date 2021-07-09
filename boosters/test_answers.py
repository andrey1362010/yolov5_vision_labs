import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from boosters.classificator.dataset import LABEL_INDEX_DICT

INPUT_PATH = "/media/andrey/big/downloads/train/test_local3.csv"
ANSWERS_PATH = "/home/andrey/IdeaProjects/yolov5/answers.csv"

input_df = pd.read_csv(INPUT_PATH)
answer_df = pd.read_csv(ANSWERS_PATH)

df = pd.merge(answer_df, input_df, left_on='frame_path', right_on='frame_path')
print("BEFORE:", df['no_gesture'][0])
#df['no_gesture'] *= 0.7
print("AFTER:", df['no_gesture'][0])

print(answer_df)
def roc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    index_roc = np.argmax(np.array(fpr) > 0.002)
    score = tpr[index_roc]
    return score

for label, index in LABEL_INDEX_DICT.items():
    if label == "no_gesture": continue
    preds = df[label].to_numpy()
    gt = (df['class'].to_numpy() == index).astype(np.float32)
    score = roc_score(gt, preds)
    for v1, v2, p in zip(gt, preds, df.frame_path):
        if v1 == 1. and v2 < 0.5: print("expected:", label, p)
    print(label, ":", score)
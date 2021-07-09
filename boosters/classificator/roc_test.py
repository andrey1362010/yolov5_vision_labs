from sklearn.metrics import roc_curve
import numpy as np
FPR_TRES = 0.002
y_true = [1, 1, 1, 0, 0, 1]
y_pred = [0.5, 0.6, 0.2, 0.3, 0.4, 0.35]
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
index_roc = np.argmax(np.array(fpr) > FPR_TRES)
score = tpr[index_roc]
print(fpr)
print(tpr)
print(index_roc, score)
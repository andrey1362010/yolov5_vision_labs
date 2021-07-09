import time

import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('tkagg')

WEIGHTS_PATH = "/home/andrey/IdeaProjects/yolov5/runs/train/exp3/weights/best.pt"
#IMAGES_PATH = "/media/andrey/big/downloads/train/1d455eb2e8131e4a2812812a684f5b19"
IMAGES_PATH = "/media/andrey/big/downloads/train/0decece6a35874910da5c72bb1515fe7" #not work...
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45

device = torch.device('cuda:0')
model = attempt_load(WEIGHTS_PATH, map_location=device)

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

dataset = LoadImages(IMAGES_PATH, img_size=imgsz, stride=stride)

#TEST RUN
model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

def get_rec(box):
    x = box[0]
    y = box[1]
    w = (box[2] - box[0])
    h = (box[3] - box[1])
    return patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

for path, img, im0s, vid_cap in tqdm(dataset):
    t = time.time()
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, 0, False, max_det=10)

    #print(1 / (time.time() - t))
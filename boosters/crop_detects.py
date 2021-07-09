import os.path
import random

import cv2
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('tkagg')

WEIGHTS_PATH = "/home/andrey/IdeaProjects/yolov5/runs/train/exp12/weights/best.pt"
IMAGES_PATH = "/media/andrey/big/downloads/train/**/*"
SAVE_PATH = "/media/andrey/big/preprocess/HANDS_CROPPED_2"

imgsz = 640
conf_thres = 0.15
iou_thres = 0.45

device = torch.device('cuda:0')
model = attempt_load(WEIGHTS_PATH, map_location=device)
model.eval()

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

def increase_box(box, image_size_x, image_size_y, zoom=1.5):
    center_x = (box[0] + box[2]) / 2.
    center_y = (box[1] + box[3]) / 2.
    w = (box[2] - box[0])
    h = (box[3] - box[1])
    new_w = w * zoom
    new_h = h * zoom
    x1 = max(center_x - new_w / 2., 0)
    y1 = max(center_y - new_h / 2., 0)
    x2 = min(center_x + new_w / 2., image_size_x)
    y2 = min(center_y + new_h / 2., image_size_y)
    return x1, y1, x2, y2

def create_box(box, image_size_x, image_size_y):
    center_x = (box[0] + box[2]) / 2.
    center_y = (box[1] + box[3]) / 2.
    w = (box[2] - box[0])
    h = (box[3] - box[1])
    new_size = max(w, h)
    x1 = max(center_x - new_size / 2., 0)
    y1 = max(center_y - new_size / 2., 0)
    x2 = min(center_x + new_size / 2., image_size_x)
    y2 = min(center_y + new_size / 2., image_size_y)
    return x1, y1, x2, y2

def random_box(image_size_x, image_size_y):
    size = random.randint(30, min(image_size_x, image_size_y) // 4.)
    x1 = random.randint(0, image_size_x - size - 1)
    y1 = random.randint(0, image_size_y - size - 1)
    x2 = x1 + size
    y2 = y1 + size
    return x1, y1, x2, y2

with torch.no_grad():
 for path, img, im0s, vid_cap in tqdm(dataset):
    image_folder = path.split('/')[-2]
    image_name = path.split('/')[-1].split(".")[0]
    save_dir_name = image_folder + "_" + image_name
    save_dir_path = os.path.join(SAVE_PATH, save_dir_name)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]

    # Apply NMS
    det = non_max_suppression(pred, conf_thres, iou_thres, 0, False, max_det=10)[0]

    #fig, ax = plt.subplots()
    #ax.imshow(im0s)

    if len(det):
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        det = det.cpu().numpy()
        for i, box in enumerate(det):
            box = increase_box(box, im0s.shape[1], im0s.shape[0])
            box = create_box(box, im0s.shape[1], im0s.shape[0])
            cropped_image = im0s[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #print(cropped_image.shape, box, im0s.shape)
            #print(os.path.join(save_dir_path, str(i) + ".jpg"))
            cv2.imwrite(os.path.join(save_dir_path, str(i) + ".jpg"), cropped_image)
            #ax.add_patch(get_rec(box))

    #plt.show()
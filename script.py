import os.path
import random
import time

import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import pandas as pd

from boosters.classificator.dataset import INDEX_LABEL_DICT, MAX_IMAGES, IMAGE_SIZE, LABEL_INDEX_DICT
from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
#matplotlib.use('tkagg')

print("start...")
WEIGHTS_PATH = "./runs/train/exp12/weights/best.pt"
WEIGHTS_PATH_CLASSIFICATOR = "./boosters/classificator/model.trc"
INPUT_PATH = 'data/test.csv'
#INPUT_PATH = '/media/andrey/big/downloads/train/test_local3.csv'
OUT_PATH = './answers.csv'

imgsz = 640
conf_thres = 0.15
iou_thres = 0.45

device = torch.device('cuda:0')
model = attempt_load(WEIGHTS_PATH, map_location=device)
model.eval()

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.final = nn.Linear(512, len(INDEX_LABEL_DICT), bias=True)
        self.attn = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, x, lens):
        x = self.normalize(x)
        x = self.features(x) #torch.Size([80, 2048, 7, 7])
        #x = x.mean(dim=(2, 3), keepdim=True)
        attn = torch.sigmoid(self.attn(x))

        x = x.permute(0, 2, 3, 1)
        attn = attn.permute(0, 2, 3, 1)
        start_index = 0
        results = []
        for l in lens:
            xp = x[start_index:start_index + l]
            ap = attn[start_index:start_index + l]
            rp = (ap * xp).sum(dim=(0, 1, 2)) / (ap.sum(dim=(0, 1, 2)) + 1e-10)
            #print(ap.shape, xp.shape, rp.shape)
            start_index += l
            results.append(rp)
        x = torch.stack(results, dim=0)
        x = self.final(x)
        return x

top_model = torchvision.models.resnet34(pretrained=False).cuda()
model_class = ResNet50Bottom(top_model).cuda()
checkpoint = torch.load(WEIGHTS_PATH_CLASSIFICATOR)
model_class.load_state_dict(checkpoint['model'])
model_class.eval()

test_df = pd.read_csv(INPUT_PATH)
paths = test_df.frame_path

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

class DetectionDataset(torch.utils.data.Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.stride = int(model.stride.max())  # model stride
        self.img_size = check_img_size(imgsz, s=stride)  # check img_size

    def __getitem__(self, index):
        path = self.paths[index]
        #print(path)
        img0 = cv2.imread(path)  # BGR
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        return img, img0, path

    def __len__(self):
        return len(self.paths)

def collate_function(batch):
    return [img for img, img0, path in batch], [img0 for img, img0, path in batch], [path for img, img0, path in batch]

dataset = DetectionDataset(paths)
test_set = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_function)

t1 = time.time()
paths_result = []
class_result = []
with torch.no_grad():
    for i, (images, images0, paths) in tqdm(enumerate(test_set, 0)):
        for path, img, im0s in zip(paths, images, images0):
            cropped_images = []
            img = torch.from_numpy(np.array(img)).to(device).float()
            img /= 255.0
            if img.ndimension() == 3: img = img.unsqueeze(0)
            pred = model(img, augment=False)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres, 0, False, max_det=12)[0]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                det = det.cpu().numpy()
                for i, box in enumerate(det):
                    box = increase_box(box, im0s.shape[1], im0s.shape[0])
                    box = create_box(box, im0s.shape[1], im0s.shape[0])
                    cropped_image = im0s[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    cropped_images.append(cropped_image)
                    #print(cropped_image, cropped_image.shape)
                    #plt.imshow(cropped_image)
                    #plt.show()

                normalized_images = []
                for image in cropped_images:
                    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                    image = np.transpose(image, (2, 0, 1))
                    image = image.astype(np.float64) / 255.
                    normalized_images.append(image)

                data = torch.from_numpy(np.array(normalized_images)).float().cuda()
                predict = model_class(data, [len(normalized_images)])
                predict = torch.softmax(predict.squeeze(0), dim=0).cpu().numpy()
            else:
                predict = np.zeros((len(LABEL_INDEX_DICT, )))
                predict[0] = 0.

            paths_result.append(path)
            class_result.append(predict)

result = {"frame_path": paths_result}
class_result = np.array(class_result)
for label, index in LABEL_INDEX_DICT.items():
    result[label] = class_result[:, index]
pd.DataFrame(result).to_csv(OUT_PATH, index=False)

print(len(test_df.frame_path) / (time.time() - t1))
import random

import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from boosters.classificator.dataset import load_labels, HandDataset, MAX_IMAGES, IMAGE_SIZE, INDEX_LABEL_DICT, \
    load_other_labels, test_train_split, collacate

LABELS_PATH = "/media/andrey/big/downloads/train.csv"
#IMAGES_OTHER_PATH = "/media/andrey/big/preprocess/HANDS_CROPPED_OTHER"
IMAGES_PATH = "/media/andrey/big/preprocess/HANDS_CROPPED_2"
labels = load_labels(IMAGES_PATH, LABELS_PATH)
#labels_other = load_other_labels(IMAGES_OTHER_PATH)

# random.shuffle(labels)
# TEST_SPLIT = 30_000
# train_labels = labels[:-TEST_SPLIT]
# test_labels  = labels[-TEST_SPLIT:]
train_labels, test_labels = test_train_split(labels)
#train_labels.extend(labels_other)

train_dataset = HandDataset(train_labels)
test_dataset  = HandDataset(test_labels, test=True)

BATCH_SIZE = 32
train_set = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True, drop_last=True, collate_fn=collacate)
test_set = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collacate)


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

def roc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    index_roc = np.argmax(np.array(fpr) > 0.002)
    score = tpr[index_roc]
    return score

def boosters_score(y_true, y_scores):
    num_classes = len(INDEX_LABEL_DICT)
    scores = []
    for j in range(1, num_classes):
        score = roc_score((y_true == j).astype(np.float64), y_scores[:, j])
        scores.append(score)
    mean_score = np.array(scores).mean()
    return mean_score


top_model = torchvision.models.resnet34(pretrained=True).cuda()
#print(list(top_model.children()))
model = ResNet50Bottom(top_model).cuda()
#print(list(model.children()))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# checkpoint = torch.load("model.trc")
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# def myce_loss(pred, label):
#     mask = np.zeros((pred.size(0), pred.size(1)))
#     for i, l in enumerate(label.detach().cpu().numpy()):
#         mask[i, l] = 1.
#     mask = torch.from_numpy(mask).float().cuda()
#
#     pred = torch.softmax(pred, dim=-1)
#     loss = - torch.sum(mask * torch.log(pred), dim=-1)
#
#     return loss.mean()
def myce_loss(pred, label):
    mask = np.zeros((pred.size(0), pred.size(1)))
    for i, l in enumerate(label.detach().cpu().numpy()):
        mask[i, l] = 1.
    mask = torch.from_numpy(mask).float().cuda()

    pred = torch.softmax(pred, dim=-1)
    no_symbol_pred = pred[:, 0]
    no_symbol_mask = mask[:, 0]
    other_symbol_pred = pred[:, 1:]
    other_symbol_mask = mask[:, 1:]
    no_symbol_loss = - no_symbol_mask * torch.log(no_symbol_pred)
    other_symbol_loss = - torch.sum(other_symbol_mask * torch.log(other_symbol_pred), dim=-1)

    LAMDA = 0.3 + 0.7 * (1. - no_symbol_pred)
    loss = no_symbol_loss + LAMDA * other_symbol_loss

    return loss.mean()

best_loss = 1000.
best_validation_loss = 0.
for epoch in range(1000000):
    print("EPOCH:", epoch)
    losses = []
    model.train()
    for i, (images, lens, classes) in enumerate(train_set, 0):

        images = torch.from_numpy(images).float().cuda()
        classes = torch.from_numpy(classes).long().cuda()
        predict = model(images, lens)

        loss = myce_loss(predict, classes)

        if i % 30 == 0:
            print(i, "/", len(train_set), "|", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #break

    mean_loss = np.array(losses).mean()
    if mean_loss < best_loss: best_loss = mean_loss
    print("MEAN LOSS:", mean_loss, "BEST:", best_loss)

    model.eval()
    with torch.no_grad():
        losses = []
        accs = []
        true_classes = []
        pred_scores  = []
        for i, (images, lens, classes) in enumerate(test_set, 0):
            images = torch.from_numpy(images).float().cuda()
            classes = torch.from_numpy(classes).long().cuda()

            predict = model(images, lens)

            loss = myce_loss(predict, classes)
            predict_indexes = np.argmax(predict.cpu().numpy(), axis=-1)
            acc = (predict_indexes == classes.cpu().numpy()).astype(np.float64).mean()
            if i % 30 == 0:
                print(i, "/", len(test_set), "| Validation:", loss.item(), "ACC:", acc)
            losses.append(loss.item())
            accs.append(acc)
            true_classes.extend(classes.cpu().numpy().tolist())
            pred_scores.extend(torch.softmax(predict, dim=-1).cpu().numpy().tolist())
            #if len(pred_scores) > 1000: break

        score = boosters_score(np.array(true_classes), np.array(pred_scores))
        mean_loss = np.array(losses).mean()
        mean_acc = np.array(accs).mean()
        print("VALIDATION LOSS:", mean_loss, "ACC:", mean_acc, "SCORE:", score)

        if score > best_validation_loss:
            print("Model SAVED...")
            best_validation_loss = score
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "model.trc")
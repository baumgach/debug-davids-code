import torch
import torch.nn as nn
from libauc.models.densenet import DenseNet121
import torch.nn.functional as F
import torchvision, torchvision.transforms
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import io
import torch.nn.functional as nnf
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pandas as pd
import torchxrayvision as xrv

# from torchvision.transforms import Transform
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def transform5(img):
    # image = cv2.imread(img, 0)
    # image = Image.fromarray(image)
    images = np.array(img)
    images = images.transpose(1, 2, 0)
    images = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
    images = cv2.resize(images, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)
    images = images / 255.0
    __mean__ = np.array([[[0.485, 0.456, 0.406]]])
    __std__ = np.array([[[0.229, 0.224, 0.225]]])
    images = (images - __mean__) / __std__
    images = images.transpose((2, 0, 1)).astype(np.float32)
    images = torch.from_numpy(images)
    return images


def transform4(img):
    print("input shape:", img.shape)
    # plt.imshow(img)
    # plt.savefig(f"/mnt/qb/work/baumgartner/djakobs46/picfir.png")
    # images = np.array(img)
    images = img.transpose(1, 2, 0)
    #    plt.imshow(images)
    #   plt.savefig(f"/mnt/qb/work/baumgartner/djakobs46/picsec.png")

    print("after np.array:", images.shape)
    images = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
    #  plt.imshow(images)
    # plt.savefig(f"/mnt/qb/work/baumgartner/djakobs46/picthird.png")
    print("after cvtColor:", images.shape)
    images = cv2.resize(images, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)
    # plt.imshow(images)
    # plt.savefig(f"/mnt/qb/work/baumgartner/djakobs46/picfourth.png")
    # print('after resize:', images.shape)
    # images = images/255.0
    # print('after normalization:', images.min(), images.max())
    # __mean__ = np.array([[[0.485, 0.456, 0.406]]])
    # __std__ =  np.array([[[0.229, 0.224, 0.225]  ]])
    # images = (images-__mean__)/__std__
    # print('after standardization:', images.min(), images.max())

    images = images.transpose((2, 0, 1)).astype(np.float32)
    print("output shape:", images.shape)
    images = torch.from_numpy(images)
    return images


Diagnosis = {
    "0": "Cardiomegaly",
    "1": "Lung Lesion",
    "2": "Edema",
    "3": "Pneumonia",
    "4": "Atelectasis",
    "5": "Pneumothorax",
    "6": "Pleural Effusion",
}
views = ["PA", "AP"]

imgpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

csvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz"
metacsvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"

transform1 = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)],
)


def resize(images, size):
    return F.interpolate(images, size=320, mode="bilinear", align_corners=False)


use_cuda = torch.cuda.is_available()

testset = xrv.datasets.MIMIC_Dataset(
    imgpath=imgpath,
    csvpath=csvpath,
    metacsvpath=metacsvpath,
    views=views,
    transform=transform1,
    unique_patients=True,
)

model = DenseNet121(pretrained=False, last_activation=None, num_classes=7)


def transform(images):
    # assert images.shape[1] == 1, "image must be gray image"
    # if images.shape == 1:
    images = cv2.cvtColor(images.numpy(), cv2.COLOR_GRAY2RGB)
    # images= cv2.cvtColor(images.numpy(), cv2.COLOR_GRAY2RGB)
    # transform1 = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)], )
    images = cv2.resize(images, (256, 256), interpolation=cv2.INTER_LINEAR)
    __mean__ = np.array([[[0.485, 0.456, 0.406]]])
    __std__ = np.array([[[0.229, 0.224, 0.225]]])
    images = (images - __mean__) / __std__
    images = images.transpose((2, 0, 1)).astype(np.float32)
    # return images
    return torch.from_numpy(images)


# def GrayToRGBAndResizeTransform(images, size=(224,224)):
#   mean = np.array([[[0.485, 0.456, 0.406]]])
#  std =  np.array([[[0.229, 0.224, 0.225]]])


# assert images.shape[0] == 1, "image must be gray image"
# images = cv2.cvtColor(images.numpy(), cv2.COLOR_GRAY2RGB)
# images = cv2.resize(images, size, interpolation=cv2.INTER_LINEAR)
# images= (images - mean) / std
# images = images.transpose((2, 0, 1)).astype(np.float32)
# return torch.from_numpy(images)
def transform_fn(img):
    return transform(img)


gray_dataset = xrv.datasets.MIMIC_Dataset(
    imgpath=imgpath,
    csvpath=csvpath,
    metacsvpath=metacsvpath,
    views=views,
    unique_patients=True,
    transform=transform4,
)
gray_dataloader = torch.utils.data.DataLoader(
    gray_dataset, batch_size=1, num_workers=0, shuffle=False
)


print(f"test dataloader size: {len(gray_dataloader)}")
print(f"whole dataset:{len(gray_dataloader.dataset)}")

Path_model1 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model123.pth"
Path_model2 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model125.pth"
Path_model3 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model127.pth"
Path_model4 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model129.pth"
Path_model5 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model131.pth"
Path_model6 = "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth"

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model1, map_location=device))

model.eval()

with torch.no_grad():
    test_pred1 = []
    true_pred1 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = images.to(device)
        y_pred1 = model(images)
        print(y_pred1)
        print(f" y_pred1 shape before sigmoid {y_pred1.shape}")
        # y_pred1 = torch.softmax(y_pred1, dim=1)
        y_pred1 = torch.sigmoid(y_pred1).detach()
        print(y_pred1)
        y_pred1 = np.squeeze(y_pred1)
        print(y_pred1)
        test_pred1.append(y_pred1.cpu().detach().numpy())
        l = np.squeeze(labels)
        true_pred1.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))
        print(f" y_pred1 shape before sigmoid {y_pred1.shape}")

print("stage 1 complete")
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model2, map_location=device))
model.eval()

with torch.no_grad():
    test_pred2 = []
    true_pred2 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = transform(images)
        images = cv2.resize(images, (224, 224), interpolation=cv2.INTER_LINEAR)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        images = (images - __mean__) / __std__
        images = images.transpose((2, 0, 1)).astype(np.float32)
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, 0)
        image = images.to(device)
        y_pred2 = model(image)
        y_pred2 = torch.sigmoid(y_pred2)
        y_pred2 = np.squeeze(y_pred2)
        test_pred2.append(y_pred2.cpu().numpy())
        l = np.squeeze(labels)
        true_pred2.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))

print("stage 2 complete")
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model3, map_location=device))
model.eval()

with torch.no_grad():
    test_pred3 = []
    true_pred3 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = transform(images)
        images = cv2.resize(images, (320, 320), interpolation=cv2.INTER_LINEAR)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        images = (images - __mean__) / __std__
        images = images.transpose((2, 0, 1)).astype(np.float32)
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, 0)
        image = images.to(device)
        y_pred3 = model(image)
        y_pred3 = torch.sigmoid(y_pred3)
        y_pred3 = np.squeeze(y_pred3)
        test_pred2.append(y_pred3.cpu().numpy())
        l = np.squeeze(labels)
        true_pred3.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))
print("stage 3 complete")
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model4, map_location=device))
model.eval()

with torch.no_grad():
    test_pred4 = []
    true_pred4 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = transform(images)
        images = cv2.resize(images, (320, 320), interpolation=cv2.INTER_LINEAR)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        images = (images - __mean__) / __std__
        images = images.transpose((2, 0, 1)).astype(np.float32)
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, 0)
        image = images.to(device)
        y_pred4 = model(image)
        y_pred4 = torch.sigmoid(y_pred4)
        y_pred4 = np.squeeze(y_pred4)
        test_pred4.append(y_pred4.cpu().numpy())
        l = np.squeeze(labels)
        true_pred4.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))
print("stage 4 complete")
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model5, map_location=device))
model.eval()

with torch.no_grad():
    test_pred5 = []
    true_pred5 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = transform(images)
        images = cv2.resize(images, (320, 320), interpolation=cv2.INTER_LINEAR)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        images = (images - __mean__) / __std__
        images = images.transpose((2, 0, 1)).astype(np.float32)
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, 0)
        image = images.to(device)
        y_pred5 = model(image)
        y_pred5 = torch.sigmoid(y_pred5)
        y_pred5 = np.squeeze(y_pred5)
        test_pred5.append(y_pred5.cpu().numpy())
        l = np.squeeze(labels)
        true_pred5.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))
print("stage 5 complete")
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(Path_model6, map_location=device))
model.eval()

with torch.no_grad():
    test_pred6 = []
    true_pred6 = []
    true_predpath = []
    for i, gray_images in enumerate(gray_dataloader):
        idx, labels, images = gray_images["idx"], gray_images["lab"], gray_images["img"]
        images = transform(images)
        images = cv2.resize(images, (320, 320), interpolation=cv2.INTER_LINEAR)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        images = (images - __mean__) / __std__
        images = images.transpose((2, 0, 1)).astype(np.float32)
        images = torch.from_numpy(images)
        images = torch.unsqueeze(images, 0)
        image = images.to(device)
        y_pred6 = model(image)
        y_pred6 = torch.sigmoid(y_pred6)
        y_pred6 = np.squeeze(y_pred6)
        test_pred6.append(y_pred6.cpu().numpy())
        l = np.squeeze(labels)
        true_pred6.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(idx))

test_pred = []
for p1, p2, p3, p4, p5, p6 in zip(
    test_pred1, test_pred2, test_pred3, test_pred4, test_pred5, test_pred6
):
    x = p1 + p2 + p3 + p4 + p5 + p6
    y = x / 6
    test_pred.append(y)

pr = pd.DataFrame(true_pred1)
# df = pd.DataFrame(true_predpath)
dl = pd.DataFrame(test_pred)
# pr.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/gtmimic.csv")
# dl.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/ensembleresultsmimic1.csv")
pr.to_csv("/mnt/qb/work/baumgartner/djakobs46/gtmimic2.csv")
dl.to_csv("/mnt/qb/work/baumgartner/djakobs46/ensemblemimic2.csv")
# df.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/picturepathseperate2.csv")

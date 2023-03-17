import torch
import torch.nn as nn
from libauc.models import DenseNet121
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import io
import torch.nn.functional as nnf
from mychexpert import CheXpert
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pandas as pd

# Path_model = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model" + ensemble + ".pth"
# data_root = '/Users/davidjakobs/Desktop/Doktorarbeit/CheXpert-v1.0-small/'
data_root = "/Users/davidjakobs/Desktop/CheXpert-v1.0-small/"
Diagnosis = {
    "0": "Cardiomegaly",
    "1": "Lung Lesion",
    "2": "Edema",
    "3": "Pneumonia",
    "4": "Atelectasis",
    "5": "Pneumothorax",
    "6": "Pleural Effusion",
}

LABELS = [
    "Cardiomegaly",
    "Lung Lesion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
]
testSet = CheXpert(
    csv_path=data_root + "valid.csv",
    image_root_path=data_root,
    use_upsampling=False,
    use_frontal=True,
    image_size=224,
    mode="valid",
    class_index=-1,
    train_cols=LABELS,
    verbose=False,
    shuffle=False,
    return_path=True,
)
testloader = torch.utils.data.DataLoader(
    testSet, batch_size=1, num_workers=0, shuffle=False
)
model = DenseNet121(pretrained=False, last_activation=None, num_classes=7)
# Path_model1 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model123.pth"
# Path_model2 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model125.pth"
# Path_model3 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model127.pth"
# Path_model4 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model129.pth"
# Path_model5 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model131.pth"
# Path_model6 = "/Users/davidjakobs/Desktop/Doktorarbeit/EnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth"
Path_model1 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model123.pth"
Path_model2 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model125.pth"
Path_model3 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model127.pth"
Path_model4 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model129.pth"
Path_model5 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model131.pth"
Path_model6 = "/Users/davidjakobs/Desktop/Doktorarbeit/NewEnsembleModel/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth"
# Path_model1 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth"
# Path_model2 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model135.pth"
# Path_model3 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model137.pth"
# Path_model4 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model139.pth"
# Path_model5 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model141.pth"
# Path_model6 = "/Users/davidjakobs/Desktop/Doktorarbeit/6diseasemodel/replicate-chexpert-resultsaucm_multi_label_pretrained_model143.pth"

device = torch.device("cpu")
model.load_state_dict(torch.load(Path_model1, map_location=device))
model.eval()
with torch.no_grad():
    test_pred1 = []
    true_pred1 = []
    true_predpath = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred1 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred1 = torch.sigmoid(y_pred1)
        y_pred1 = np.squeeze(y_pred1)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred1.append("Probability for patient " + str(jdx) + ":" + str(y_pred1.cpu().detach().numpy()))
        test_pred1.append(y_pred1.cpu().detach().numpy())
        l = np.squeeze(test_labels)
        true_pred1.append(l.numpy())
        true_predpath.append(str(l.numpy()) + str(test_path))
        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
model.load_state_dict(torch.load(Path_model2, map_location=device))
model.eval()
with torch.no_grad():
    test_pred2 = []
    true_pred2 = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred2 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred2 = torch.sigmoid(y_pred2)
        y_pred2 = np.squeeze(y_pred2)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred2.append("Probability for patient " + str(jdx) + ":" + str(y_pred2.cpu().detach().numpy()))
        test_pred2.append(y_pred2.cpu().detach().numpy())
        #     true_pred2.append(test_labels.numpy())
        true_pred2.append(test_path)
        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
model.load_state_dict(torch.load(Path_model3, map_location=device))
model.eval()
with torch.no_grad():
    test_pred3 = []
    true_pred3 = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred3 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred3 = torch.sigmoid(y_pred3)
        y_pred3 = np.squeeze(y_pred3)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred3.append("Probability for patient " + str(jdx) + ":" + str(y_pred3.cpu().detach().numpy()))
        test_pred3.append(y_pred3.cpu().detach().numpy())
        #     true_pred3.append(test_labels.numpy())
        true_pred3.append(test_path)
        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
model.load_state_dict(torch.load(Path_model4, map_location=device))
model.eval()
with torch.no_grad():
    test_pred4 = []
    true_pred4 = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred4 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred4 = torch.sigmoid(y_pred4)
        y_pred4 = np.squeeze(y_pred4)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred4.append("Probability for patient " + str(jdx) + ":" + str(y_pred4.cpu().detach().numpy()))
        test_pred4.append(y_pred4.cpu().detach().numpy())
        #     true_pred4.append(test_labels.numpy())
        true_pred4.append(test_path)
        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
model.load_state_dict(torch.load(Path_model5, map_location=device))
model.eval()
with torch.no_grad():
    test_pred5 = []
    true_pred5 = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred5 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred5 = torch.sigmoid(y_pred5)
        y_pred5 = np.squeeze(y_pred5)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred5.append("Probability for patient " + str(jdx) + ":" + str(y_pred5.cpu().detach().numpy()))
        test_pred5.append(y_pred5.cpu().detach().numpy())
        #     true_pred5.append(test_labels.numpy())
        true_pred5.append(test_path)
        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
model.load_state_dict(torch.load(Path_model6, map_location=device))
model.eval()
with torch.no_grad():
    test_pred6 = []
    true_pred6 = []
    for jdx, data in enumerate(testloader, 64541):
        test_data, test_labels, test_path = data
        test_data = test_data.to(device)
        y_pred6 = model(test_data)
        # pred= np.argmax(y_pred)
        y_pred6 = torch.sigmoid(y_pred6)
        y_pred6 = np.squeeze(y_pred6)
        # _, y_hat = y_pred.max(1)
        # predicted_idx = str(y_hat.item())
        # dict = Diagnosis[predicted_idx]
        # test_pred6.append("Probability for patient " + str(jdx) + ":" + str(y_pred6.cpu().detach().numpy()))
        test_pred6.append(y_pred6.cpu().detach().numpy())
        true_pred6.append(test_path)

        # max_pred.append("Prediction for patient " + str(jdx) +": "  + dict)
test_pred = []
for p1, p2, p3, p4, p5, p6 in zip(
    test_pred1, test_pred2, test_pred3, test_pred4, test_pred5, test_pred6
):
    #  test_pred.append(p1+p2+p3+p4+p5+p6)
    x = p1 + p2 + p3 + p4 + p5 + p6
    y = x / 6
    test_pred.append(y)

# print(true_pred1)
# print(true_predpath)
# print(true_pred2)
# print(true_pred3)
# print(true_pred4)
# print(true_pred5)
# print(true_pred6)
# print(test_pred6)

# print(np.shape(test_pred))
# print(np.shape(test_pred6))
# print(np.shape(finallist))
# print(max_pred)
# np.squeeze(test_pred1)
pr = pd.DataFrame(true_pred1)
df = pd.DataFrame(true_predpath)
dl = pd.DataFrame(test_pred)
pr.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/gtnew.csv")
dl.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/ensembleresultsnew2.csv")
df.to_csv("/Users/davidjakobs/Desktop/Doktorarbeit/picturepathseperate2.csv")

from libauc.models.densenet import DenseNet121
import torchxrayvision as xrv
import torchvision
import torch
from mychexpert import CheXpert
from mymimic import MIMIC_Dataset
import matplotlib.pyplot as plt

def make_mimic_data():

    imgpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
    csvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    metacsvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
    views = ["PA", "AP"]

    mimic_data = MIMIC_Dataset(
        imgpath=imgpath,
        csvpath=csvpath,
        metacsvpath=metacsvpath,
        views=views,
        unique_patients=True,
        image_size=224, 
        return_path=True,
    )
    
    return mimic_data

def make_chexpert_data():
    
    data_root = "/mnt/qb/baumgartner/rawdata/CheXpert/CheXpert-v1.0-small/"
    LABELS = [
        "Cardiomegaly",
        "Lung Lesion",
        "Edema",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
    ]
    
    chexpert_data = CheXpert(
        csv_path=data_root+'valid.csv',  
        image_root_path=data_root, 
        use_upsampling=False, 
        use_frontal=True, 
        image_size=224, 
        mode='valid', 
        class_index=-1, 
        train_cols=LABELS, 
        verbose=False, 
        shuffle=False, 
        return_path=True
    )
    
    return chexpert_data

mimic_data = make_mimic_data()
chexpert_data = make_chexpert_data()

model_paths = [
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model123.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model125.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model127.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model129.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model131.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth",
]

model = DenseNet121(pretrained=False, last_activation=None, num_classes=7)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(model_paths[0], map_location=device))
model.eval()

mimic_loader = torch.utils.data.DataLoader(
    mimic_data, batch_size=1, num_workers=0, shuffle=False
)
mimic_iter = iter(mimic_loader)

chexpert_loader = torch.utils.data.DataLoader(
    chexpert_data, batch_size=1, num_workers=0, shuffle=False
)
chexpert_iter = iter(chexpert_loader)

xm, ym, _ = next(mimic_iter)
xc, yc, _ = next(chexpert_iter)

print('MIMIC', xm.shape, xm.min(), xm.max())
print('CheXpert', xc.shape, xc.min(), xc.max())

plt.subplot(121)
plt.imshow(xc.squeeze()[0])
plt.title('CheXpert')
plt.subplot(122)
plt.imshow(xm.squeeze()[0])
plt.title('MIMIC')
plt.show()

xm, xc = xm.cuda(), xc.cuda()

print('MIMIC')
y = model(xm)
print(torch.sigmoid(y))
print(ym)
print('--')
print('Chexpert')
y = model(xc)
print(torch.sigmoid(y))
print(yc)
from libauc.models.densenet import DenseNet121
import torchxrayvision as xrv
import torchvision
import torch

model = DenseNet121(pretrained=False, last_activation=None, num_classes=7)

imgpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
csvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz"
metacsvpath = "/mnt/qb/baumgartner/rawdata/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
views = ["PA", "AP"]

transform1 = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)],
)

testset = xrv.datasets.MIMIC_Dataset(
    imgpath=imgpath,
    csvpath=csvpath,
    metacsvpath=metacsvpath,
    views=views,
    transform=transform1,
    unique_patients=True,
)

model_paths = [
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model123.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model125.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model127.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model129.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model131.pth",
    "/mnt/qb/work/baumgartner/djakobs46/replicate-chexpert-resultsaucm_multi_label_pretrained_model133.pth",
]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
model.load_state_dict(torch.load(model_paths[0], map_location=device))

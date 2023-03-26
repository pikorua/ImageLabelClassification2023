import os

import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

# Define a transform to convert PIL
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor(),


])


transform2 = torchvision.transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dirName = "..\images\\"
for filename in os.listdir(dirName):
    # Read a PIL image
    image = Image.open(dirName + filename).convert('RGB')

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image).float()
    img_tensor = transform2(img_tensor)

    saveName = filename[2:-4]
    torch.save(img_tensor, ".\..\ptImages\\" + saveName + ".pt")

for i in range(20000):
    res = torch.zeros(14)
    torch.save(res, ".\..\labels\\" + str(i+1) + ".pt")


labelDir = "..\\annot\\"

for filename in os.listdir(labelDir):
    indexOfLabel = int(filename[:-4])
    file = open(labelDir + filename)
    for line in file:
        tmp = torch.load("..\\labels\\" + line.strip() + ".pt")
        tmp[indexOfLabel] = 1
        savepath = "..\\labels\\" + line.strip() + ".pt"
        torch.save(tmp, savepath)
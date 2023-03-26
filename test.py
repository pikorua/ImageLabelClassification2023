import csv
import os

import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

# Define a transform to convert PIL
# image to a Torch tensor
from src.main import MultiLabelResNet

transform = transforms.Compose([
    transforms.PILToTensor(),


])


transform2 = torchvision.transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dirName = ".\imagesTest\images\\"
model = MultiLabelResNet(num_classes=14)
model.load_state_dict(torch.load("./src/resnetMultiLabelNet.pt"))
model.eval()
with open('testResults.tsv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for filename in os.listdir(dirName):
        # Read a PIL image
        image = Image.open(dirName + filename).convert('RGB')

        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        img_tensor = transform(image).float()
        img_tensor = transform2(img_tensor)

        saveName = filename[2:-4]
        #torch.save(img_tensor, ".\..\ptImagesTest\\" + saveName + ".pt")
        output = model(img_tensor.unsqueeze(0))
        pred = np.where(output > 0.4, 1, 0)
        listOfPred = list(pred.flatten())
        toStr = list(map(str, listOfPred))
        listOfPred = [saveName] + toStr

      # specify tab as the delimiter
        writer.writerow(listOfPred)  # write each row to the file
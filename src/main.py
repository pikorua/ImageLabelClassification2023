import numpy as np
import sklearn.metrics
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
from src.multiLabelDataset import customDataset


batch_size = 100
num_classes = 14
num_epochs = 30
threshold = 0.4

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load the ResNet-18 model pre-trained on ImageNet
        self.resnet = models.resnet18(pretrained=True)

        # Replace the last fully connected layer with a new one that outputs the correct number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-18 model
        x = self.resnet(x)

        # Apply sigmoid activation to each output to obtain the probability of each class
        #x = F.sigmoid(x)

        return x





if __name__ == "__main__":

    full_data = customDataset(label_dir="../labels", image_dir="../ptImages")
    train_data_size = int(0.8 * len(full_data))
    valid_data_size = int(0.2 * len(full_data))
    assert train_data_size + valid_data_size == len(full_data)
    train_data, valid_data = torch.utils.data.random_split(full_data, [train_data_size, valid_data_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)


    net = MultiLabelResNet(num_classes)
# Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    prevAccuracy = 0
# Train the neural network on the dataset
    for epoch in range(num_epochs):
        prevLoss = 100
        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = sample['image'].float()
            y = sample['label']

            # Forward pass through the neural network
            outputs = net(X)

            # Compute the loss and perform backpropagation
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()



            print("epoch: " + str(epoch) + " batch number: " + str(i) + " with loss: " + str(loss.item()))

        # Evaluate the ne ural network on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            i = 0
            sumprec = 0
            sumrec = 0
            sumf1 = 0
            for sample in valid_dataloader:
                X = sample['image'].float()
                y = sample['label']

                outputs = net(X)
                pred = np.where(outputs > threshold, 1, 0)
                prec, rec, f1, support = sklearn.metrics.precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
                sumprec += prec
                sumrec += rec
                sumf1 += f1

                outputs = outputs * 1
                predicted = (outputs > threshold).float()
                total += y.size(0)
                correct += (predicted == y).all(dim=1).sum().item()
                i += 1
                print(i)
        print("Precision: " + str(sumprec/i))
        print("Recall: " + str(sumrec/i))
        print("F1: " + str(sumf1/i))
        accuracy = 100 * correct / total
        print("Accuracy: {:.2f}%".format(accuracy))
        if (accuracy < prevAccuracy):
            break
        else:
            prevAccuracy = accuracy
            torch.save(net.state_dict(), "./resnetMultiLabelNet.pt")

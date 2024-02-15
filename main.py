# -*- coding = utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1) # set the random seed

#Convolutional Neural Network Architecture
class CNN_MNISTClassifier(nn.Module):
    def __init__(self):
        super(CNN_MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(64, 64, 4) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv3 = nn.Conv2d(64, 64, 2) #in_channels, out_chanels, kernel_size
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    print('Convolutional Neural Network Architecture Done')

class CNN_MNISTClassifier_2(nn.Module):
    def __init__(self):
        super(CNN_MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(64, 64, 4) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv3 = nn.Conv2d(64, 64, 2) #in_channels, out_chanels, kernel_size
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        y = self.pool(F.relu(self.conv1(y)))
        y = self.pool(F.relu(self.conv2(y)))
        y = self.pool(F.relu(self.conv3(y)))
        z = torch.cat((x,y),dim=1)
        z = z.view(-1, 160)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z
    
    print('Convolutional Neural Network Architecture Done')

if __name__ == '__main__':
    print("CNN")
    model = CNN_MNISTClassifier()
    print(model)
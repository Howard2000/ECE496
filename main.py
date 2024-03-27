# -*- coding = utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent
from torch.utils.data import Dataset
import mat4py

torch.manual_seed(1) # set the random seed

class RatDataset(Dataset):

    def __init__(self, train = True):
        self.fold = [[],[],[]]

        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat4Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat5Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat6Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat7Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat8Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat9Training_Fold1.mat'))
        self.fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat10Training_Fold1.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat4Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat5Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat6Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat7Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat8Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat9Training_Fold2.mat'))
        self.fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat10Training_Fold2.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat4Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat5Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat6Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat7Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat8Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat9Training_Fold3.mat'))
        self.fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/ECE496/7x8/Rat10Training_Fold3.mat'))

        self.data = torch.empty(0,5600)
        self.labels = torch.empty(0)

        for i in range (3):
            for j in range (7):
                if train:
                    train_data = torch.tensor(self.fold[i][j]['training_data_rat']).transpose(1,0)
                    self.data = torch.cat((self.data, train_data), 0)

                    train_lables = torch.tensor(self.fold[i][j]['training_data_labels']).squeeze(1)
                    self.labels = torch.cat((self.labels, train_lables), 0)

                else:
                    test_data = torch.tensor(self.fold[i][j]['test_data_rat']).transpose(1,0)
                    self.data = torch.cat((self.data, test_data), 0)

                    test_labels = torch.tensor(self.fold[i][j]['test_data_labels']).squeeze(1)
                    self.labels = torch.cat((self.labels, test_labels), 0)


    def __len__(self):
        return torch.numel(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.label[idx]
    

#Convolutional Neural Network Architecture
class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
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

class CNN_Classifier_2(nn.Module):
    def __init__(self):
        super(CNN_Classifier_2, self).__init__()
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

def get_accuracy(model, data):
    device = torch.device('cuda:0')   
    kwargs = {'num_workers': 1, 'pin_memory': True}
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64, **kwargs):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=1):
    device = torch.device('cuda:0')   
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)
    test_data = RatDataset(False)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)             # forward pass

            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, data)) # compute training accuracy
            val_acc.append(get_accuracy(model, test_data))  # compute validation accuracy
            n += 1

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

if __name__ == '__main__':
    device = torch.device('cuda:0')
    print("CNN")
    model = CNN_Classifier().to(device)
    data = RatDataset(train = True)
    train(model, data)
    print(model)
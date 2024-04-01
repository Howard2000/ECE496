# -*- coding = utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent
from torch.utils.data import Dataset
import mat4py

class RatDataset(Dataset):

    def __init__(self, train = True):
        
        self.data = torch.empty(0,5600)
        self.labels = torch.empty(0)

        for i in range (3):
            for j in range (7):
                if train:
                    train_data = torch.tensor(fold[i][j]['training_data_rat']).transpose(1,0)
                    self.data = torch.cat((self.data, train_data), 0)

                    train_lables = torch.tensor(fold[i][j]['training_data_labels']).squeeze(1)
                    train_lables = torch.sub(train_lables, torch.ones_like(train_lables))
                    self.labels = torch.cat((self.labels, train_lables), 0)

                else:
                    test_data = torch.tensor(fold[i][j]['test_data_rat']).transpose(1,0)
                    self.data = torch.cat((self.data, test_data), 0)

                    test_labels = torch.tensor(fold[i][j]['test_data_labels']).squeeze(1)
                    test_labels = torch.sub(test_labels, torch.ones_like(test_labels))

                    self.labels = torch.cat((self.labels, test_labels), 0)
        print("Data done")

    def __len__(self):
        return torch.numel(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.labels[idx]
    

#Convolutional Neural Network Architecture
class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 8) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(64, 64, 4) #in_channels, out_chanels, kernel_size
        self.pool2 = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv3 = nn.Conv2d(64, 64, 2) #in_channels, out_chanels, kernel_size
        self.pool3 = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)
        print('Convolutional Neural Network Architecture Done')


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class CNN_Classifier_2(nn.Module):
    def __init__(self):
        super(CNN_Classifier_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 8) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(64, 64, 4) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv3 = nn.Conv2d(64, 64, 2) #in_channels, out_chanels, kernel_size
        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 3)
        print('Convolutional Neural Network 2 Architecture Done')


    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        y = self.pool(F.relu(self.conv1(y)))
        y = self.pool(F.relu(self.conv2(y)))
        y = self.pool(F.relu(self.conv3(y)))
        z = torch.cat((x,y),dim=1)
        z = z.view(64, -1)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z
    
def get_accuracy(model, data):
    device = torch.device('cuda:0')   
    kwargs = {'num_workers': 2, 'pin_memory': False}
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64, **kwargs):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(torch.reshape(imgs, (-1,1,100,56)))

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=1):
    device = torch.device('cuda:0')
    # torch.set_default_device('cuda:0')
    kwargs = {'num_workers': 2, 'pin_memory': True}
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
            if (n % 500 == 0 ):
                print(f"TRaining {n}")
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()         # a clean up step for PyTorch
            out = model(torch.reshape(imgs, (-1,1,100,56))).to(device)         
            labels = labels.type(torch.LongTensor).to(device)    # forward pass

            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, train=True))
            val_acc.append(get_accuracy(model, train=False))

            n += 1
        print("Epoch {} Training Accuracy: {}".format(epoch, get_accuracy(model, data)))
        print("Epoch {} Validation Accuracy: {}".format(epoch, get_accuracy(model, test_data)))
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


    print("Final Training Accuracy: {}".format(get_accuracy(model, data)))
    print("Final Validation Accuracy: {}".format(get_accuracy(model, test_data)))

if __name__ == '__main__':
    torch.manual_seed(1) # set the random seed
    fold = [[],[],[]]

    fold[0].append(mat4py.loadmat('./data/Rat4Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold1.mat'))
    # fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold1.mat'))
    print("Done ")
    fold[1].append(mat4py.loadmat('./data/Rat4Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold2.mat'))
    # fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold2.mat'))
    print("Done ")
    # fold[2].append(mat4py.loadmat('./data/Rat4Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold3.mat'))
    # fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold3.mat'))
    print("Done ")
    device = torch.device('cuda:0')
    print("CNN")
    model = CNN_Classifier().to(device)
    data = RatDataset(train = True)
    train(model, data, num_epochs=100)
    # print(model)
# 

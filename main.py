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
                    # print(f"train_Data shape: {train_data.size()}")

                    self.data = torch.cat((self.data, train_data), 0)

                    train_lables = torch.tensor(fold[i][j]['training_data_labels']).squeeze(1)
                    # print(f"train_lables.shape: {train_lables.size()}")
   
                    train_lables = torch.sub(train_lables, torch.ones_like(train_lables))
                    self.labels = torch.cat((self.labels, train_lables), 0)

                else:
                    test_data = torch.tensor(fold[i][j]['test_data_rat']).transpose(1,0)
                    self.data = torch.cat((self.data, test_data), 0)

                    test_labels = torch.tensor(fold[i][j]['test_data_labels']).squeeze(1)
                    test_labels = torch.sub(test_labels, torch.ones_like(test_labels))

                    self.labels = torch.cat((self.labels, test_labels), 0)
        print("Data done")
        idx = torch.randperm(self.data.shape[0])
        self.data = self.data[idx].view(self.data.size())
        self.labels = self.labels[idx].view(self.labels.size())

        print(f"shape: {self.data.size()} lables: {self.labels.size()}")

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
        self.conv3 = nn.Conv2d(64, 64, 2) #in_channels, out_chanels, kernel_size
        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 3)
        print('Convolutional Neural Network Architecture Done')


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    n = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64, **kwargs):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(torch.reshape(imgs, (-1,1,56,100)))

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
        # if (n == 0):
        #     print(f"output shape: {output.size()} pred shape: {pred.size()} lables: {labels.size()}")
        #     print("Output: ")
        #     print(output[0:10])
        #     print("Pred: ")
        #     print(pred[0:10])
        #     print("lables :")
        #     print(labels[0:10])
        n += 1
    return correct / total

def train(model, data, batch_size=64, num_epochs=1):
    device = torch.device('cuda:0')
    # torch.set_default_device('cuda:0')
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)
    test_data = RatDataset(False)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.8)
    train_acc = []
    val_acc = []
    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()         # a clean up step for PyTorch
            out = model(torch.reshape(imgs, (-1,1,56,100))).to(device)         
            labels = labels.type(torch.LongTensor).to(device)    # forward pass

            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            

            # save the current training information
            
            losses.append(float(loss)/batch_size)             # compute *average* loss
            if (n % 2000 == 0 ):
                print(f"TRaining {n}")
                # print(f"out.shape: {out.size()}")
                # print(f"lables shape: {labels.size()}")
            
            n += 1
        iters.append(epoch)
        train = get_accuracy(model, data)
        val = get_accuracy(model, test_data)
        train_acc.append(train)
        val_acc.append(val)
        print("Epoch {} Training Accuracy: {}".format(epoch, train))
        print("Epoch {} Validation Accuracy: {}".format(epoch, val))
    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.title("Acc Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()



    print("Final Training Accuracy: {}".format(get_accuracy(model, data)))
    print("Final Validation Accuracy: {}".format(get_accuracy(model, test_data)))

if __name__ == '__main__':
    torch.manual_seed(1) # set the random seed
    fold = [[],[],[]]

    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat4Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold1.mat'))
    fold[0].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold1.mat'))
    print("Done ")
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat4Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold2.mat'))
    fold[1].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold2.mat'))
    print("Done ")
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat4Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat5Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat6Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat7Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat8Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat9Training_Fold3.mat'))
    fold[2].append(mat4py.loadmat('C:/Users/pablo/Documents/Capstone/7x8/Rat10Training_Fold3.mat'))
    print(len(fold[0][0]))
    print("Done ")
    device = torch.device('cuda:0')
    print("CNN")
    model = CNN_Classifier().to(device)
    data = RatDataset(train = True)
    train(model, data, num_epochs=50)
    # print(model)
# 

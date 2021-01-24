import numpy as np
import torch
import torchvision
import torch.nn as nn
import h5py
import time
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


transform1 = transforms.Compose([transforms.RandomVerticalFlip(p = 0.1), 
                                transforms.RandomHorizontalFlip(p = 0.1), transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform1)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform2)

class BasicBlock(nn.Module):
    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.Bnorm1 = nn.BatchNorm2d(dim)
        self.Bnorm2 = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = F.relu(self.Bnorm1(x))
        x = self.Bnorm2(self.conv(x))
        x += residual
        x = F.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, dim):
        super(Bottleneck, self).__init__()
        self.conv_1 = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1)
        self.conv_r = nn.Conv2d(dim, 2*dim, kernel_size=1, stride=2, padding=0)
        self.Bnorm1 = nn.BatchNorm2d(dim*2)
        self.Bnorm2 = nn.BatchNorm2d(dim*2)
    
    def forward(self, x):
        residual = x
        residual = self.conv_r(residual)
        x = self.conv_1(x)
        x = F.relu(self.Bnorm1(x))
        x = self.Bnorm2(self.conv_2(x))
        x += residual
        x = F.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = BasicBlock(32)
        self.layer3_1 = Bottleneck(32)
        self.layer3_2 = BasicBlock(64)
        self.layer4_1 = Bottleneck(64)
        self.layer4_2 = BasicBlock(128)
        self.layer5_1 = Bottleneck(128)
        self.layer5_2 = BasicBlock(256)
        
        # dropout
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Bnorm1 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(256, 500)

    def forward(self, x):
        # first layer
        x = self.conv(x)
        x = F.relu(self.Bnorm1(x))
        x = self.dropout1(x)
        # second layer
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        # third layer
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_2(x)
        x = self.layer3_2(x)
        x = self.dropout3(x)
        # forth layer
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_2(x)
        x = self.layer4_2(x)
        x = self.dropout4(x)
        # fifth layer
        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.maxpool(x)
        
        x = x.view(-1, 256)
        x = self.dropout5(x)
        x = self.fc(x)
        return x
model = Net()
model.cuda()

LR = 0.001
batch_size = 500
Num_Epochs = 100

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = LR)

for epoch in range(Num_Epochs):
    time1 = time.time()
    model.train()
    for i, (images, classes) in enumerate(train_loader):
        data, target = Variable(images.cuda()), Variable(classes.cuda())
#         data, target = Variable(images), Variable(classes)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    model.eval()

    # Test Loss
    counter = 0
    test_accuracy_sum = 0.0
    for i, (images, classes) in enumerate(test_loader):
        data, target = Variable(images.cuda()), Variable(classes.cuda())
#         data, target = Variable(images), Variable(classes)
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum/float(counter)
    time2 = time.time()
    time_elapsed = time2 - time1
    print(epoch, test_accuracy_ave, time_elapsed)
    
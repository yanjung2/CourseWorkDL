import numpy as np
import torch
import torchvision
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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # dropout
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.dropout5 = torch.nn.Dropout(p=0.5)
        self.dropout6 = torch.nn.Dropout(p=0.5)
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch Normalization
        self.Bnorm1 = torch.nn.BatchNorm2d(64)
        self.Bnorm2 = torch.nn.BatchNorm2d(64)
        self.Bnorm3 = torch.nn.BatchNorm2d(64)
        self.Bnorm4 = torch.nn.BatchNorm2d(64)
        self.Bnorm5 = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(4*4*64, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc3 = torch.nn.Linear(500, 10)

    def forward(self, x):
        
        x = self.Bnorm1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        p = self.dropout1(self.pool(x))
        
        x = self.Bnorm2(F.relu(self.conv3(p)))
        x = F.relu(self.conv4(x))
        p = self.dropout2(self.pool(x))
       
        x = self.Bnorm3(F.relu(self.conv5(p)))
        x = self.dropout3(F.relu(self.conv6(x)))
        x = self.Bnorm4(F.relu(self.conv7(x)))
        x = self.Bnorm5(F.relu(self.conv8(x)))

        x = self.dropout4(x)
        
        x = x.view(-1, 4*4*64)

        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
model = Net()
model.cuda()

LR = 0.001
batch_size = 500
Num_Epochs = 50

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = LR)

for epoch in range(Num_Epochs):
    time1 = time.time()
    model.train()
    for i, (images, classes) in enumerate(train_loader):
        data, target = Variable(images.cuda()), Variable(classes.cuda())
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
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum/float(counter)
    time2 = time.time()
    time_elapsed = time2 - time1
    print(epoch, test_accuracy_ave, time_elapsed)
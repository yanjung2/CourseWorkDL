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
import torchvision.models as models

DIM = 224

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(DIM, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform_test)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,100)
# model = Net()
model.cuda()

LR = 0.001
batch_size = 100
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
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)
            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4(h)
        h = model.avgpool(h)
        h = h.view(h.size(0), -1)
        output = model.fc(h)
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
    
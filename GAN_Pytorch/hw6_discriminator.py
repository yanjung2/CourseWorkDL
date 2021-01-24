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

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196, 32, 32)),
            nn.LeakyReLU(0.1),

            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196, 16, 16)),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196, 16, 16)),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196, 8, 8)),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196, 8, 8)),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196, 8, 8)),
            nn.LeakyReLU(0.1),         
            
            nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((196, 8, 8)),
            nn.LeakyReLU(0.1),
                        
            nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm((196, 4, 4)),
            nn.LeakyReLU(0.1),
            
            nn.MaxPool2d(kernel_size=4, stride=4)       
        )
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, 196)
        x = self.fc10(x)
        return x
        
        
learning_rate = 0.0001
batch_size = 128
Num_Epochs = 100

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model = discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(Num_Epochs):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000
    time1 = time.time()
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    model.train()
    for i, (images, classes) in enumerate(train_loader):
        if(classes.shape[0] < batch_size):
            continue
        data, target = Variable(images).cuda(), Variable(classes).cuda()
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
        data, target = Variable(images).cuda(), Variable(classes).cuda()
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
    
torch.save(model,'cifar10.model')
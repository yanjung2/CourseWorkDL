import numpy as np
import torch
import torchvision
import torch.nn as nn
import h5py
import time
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.models as models


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
    def forward(self, input, extract_features=0):
        x = self.main(input)
        x = x.view(-1, 196)
        if(extract_features==8):
            return x
        x = self.fc10(x)
        return x

batch_size = 128
    
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('cifar10.model')
model.cuda()
model.eval()

j=0
for batch_idx, (X_batch, Y_batch) in testloader:
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1)%10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()
    if j == 0:
        break

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import os

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in xrange(200):
    _, output = model(X)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features.png', bbox_inches='tight')
plt.close(fig)
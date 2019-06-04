import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #128X128
        self.convolucionalOne = nn.Conv2d(3,10,5)
        #124X124
        self.pool = nn.MaxPool2d(2,2)
        #62X62
        self.convolucionalTwo = nn.Conv2d(10,20,5)
        #58X58
        #29X29
        self.funcionLinealOne = nn.Linear(20*29*29,450)
        self.funcionLinealTwo = nn.Linear(450,100)

    def forward(self,x):
        x = self.convolucionalOne(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.convolucionalTwo(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,20*29*29)
        x = self.funcionLinealOne(x)
        x = F.relu(x)
        x = self.funcionLinealTwo(x)

        return F.log_softmax(x,dim=1)
    

        


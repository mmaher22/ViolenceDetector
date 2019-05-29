#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from model import *
from dataloader import *

#train function 
def train(my_dataloader,epochs,model,lf,optimizer):
    # Implement no of epochs
    for epoch in range(epochs):
        # 
        for (batch, (inp, targ)) in enumerate(my_dataloader):
            model.zero_grad()
            inp = inp.float()
            targ = targ.float()
            pred = model(inp)
            loss = lf(pred,targ)
            loss.backward()
            optimizer.step()
        print('Epoch:{} loss: {}'.format(epoch,loss.data.numpy()))

# function to check code over validation set
def test(val_dataloader,model):
    correct = 0
    total = 0
    for (batch, (inp, targ)) in enumerate(val_dataloader):
        inp = inp.float()
        targ = targ.numpy()
        targ = targ.flatten()
        pred =model(inp)
        pred = pred.detach().numpy()
        pred = pred.flatten()
        pred = np.where(pred > 0.5, 1, 0)
        correct += np.sum(pred==targ)
        total += pred.shape[0]
    print(correct/total)


if __name__ == '__main__':
    BATCH_SIZE =16
    inpu,targ = data_loader()
    my_dataloader,val_dataloader = load_data(inpu,targ,BATCH_SIZE)
    experiment parameters
    epochs =30
    units = 256 # No of Lstm units
    dimensions = 144 # Dimensions of the feature used
    lf = nn.BCELoss() # loss function to compare each predicted output    
    model = Model(units,dimensions)
    optimizer = torch.optim.Adam(list(model.parameters()),lr=0.00001)  
    train(my_dataloader,epochs,model,lf,optimizer)
    test(val_dataloader,model)

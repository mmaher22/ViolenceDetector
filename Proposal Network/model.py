
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import numpy as np
import torch

class Model(nn.Module):
    def __init__(self,units,feature_size):
        super(Model, self).__init__()
        self.lstm =  nn.LSTM(input_size =feature_size , hidden_size = units,batch_first=True,bidirectional=True) 
        self.layer1 = nn.Linear(units,1)
        self.layer2 = nn.Linear(units,1)
        self.sf = nn.Sigmoid()
        
    def forward(self,inp):
        # Features put in Output from Bidirectional LSTM
        output,_ = self.lstm(inp)
        # Extract output1 for forward part of bidirectional lstm, use a fc layer and sigmoid output for score
        out1 = self.sf(self.layer1(output[:,:,:units]))
        # Extract output2 for reverse part of bidirectional lstm, use a fc layer and sigmoid output for score
        out2 = self.sf(self.layer2(output[:,:,-units:]))
        # combine the two scores to get the final score
        res = out1*out2
        res = torch.squeeze(res)
        return res        

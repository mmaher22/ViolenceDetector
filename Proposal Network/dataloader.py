
from scipy.io import loadmat
import h5py 
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch.utils.data as utils
import torch

def check_range(no,rang):
    if(rang[0] <= no <= rang[1]):
        return True
    return False

# Data loader
def data_loader():
    features = 'VSD_2014_December_official_release/YouTube-gen/features/'
    ann = 'VSD_2014_December_official_release/YouTube-gen/annotations/'
    inpu = []
    targ =[]
    # extract features from the diretory
    for ind,file in enumerate(os.listdir(features)):
        sec_part = file.split('_')[1]
        first = file.split('_')[0]
        ftype = sec_part.split('.')[0]
        # consider only visual feature files
        if(ftype == 'visual'):
            try:
                x = h5py.File(features+file,'r')
            except:
                continue
            feats = x['LBP']# select the type of features here. 
            intrs=[]
            f = open(ann+first+'.txt','r')
            for i,line in enumerate(f):
                inter = np.array(line.strip().split(' ')).astype(np.int)
                if(inter[0]+150 > len(feats)):
                    continue
                try:    
                    inpu.append(feats[inter[0]:inter[0]+150])
                except:
                    continue
                tar = np.zeros(150)
                if(inter[0] +150 > inter[1]):
                    tar[:inter[1]-inter[0]]=1
                    targ.append(tar)
                else:
                    tar[:]=1
                    targ.append(tar)
                intrs.append(inter)
            for t in range(5):
                for inte in intrs:
                    try:
                        starting = np.random.randint(0,len(feats)-151)
                    except:
                        continue
                    inpu.append(feats[starting:starting+150])
                    tar = np.zeros(150)
                    for ints in intrs:
                        if(check_range(starting,ints) and not check_range(starting+150,ints)):
                            tar[:ints[1]-starting]=1
                        elif(check_range(starting+150,ints) and not check_range(starting,ints)):
                            tar[-(starting+150-ints[0]):]=1
                        elif(check_range(starting+150,ints) and check_range(starting,ints)):
                            tar[:]=1
                    targ.append(tar)
    inpu = np.array(inpu).astype(np.float)
    targ = np.array(targ)
    return inpu,targ


def load_data(BATCH_SIZE):
    inpu,targ = data_loader()
    #create the test-train split here
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(inpu, targ, test_size=0.3)
    # convert numpy arrays to pytorch stack
    tensor_x = torch.stack([torch.from_numpy(i) for i in input_tensor_train]) 
    tensor_y = torch.stack([torch.from_numpy(i) for i in target_tensor_train])
    # convert numpy arrays to pytorch stach validation set
    val_x = torch.stack([torch.from_numpy(i) for i in input_tensor_val])
    val_y = torch.stack([torch.from_numpy(i) for i in target_tensor_val])
    # create train and test validation set
    my_dataset = utils.TensorDataset(tensor_x,tensor_y) 
    my_dataloader = utils.DataLoader(my_dataset,batch_size=BATCH_SIZE)
    val_dataset = utils.TensorDataset(val_x,val_y) 
    val_dataloader = utils.DataLoader(val_dataset,batch_size=BATCH_SIZE)
    return my_dataloader,val_dataloader
    
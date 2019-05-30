import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class makeDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, spatialTransform):
        self.spatial_transform = spatialTransform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.seqLen = 20

    def __getitem__(self, idx):
        #Hog Features Extractor Parameters
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)
        
        vidName = self.images[idx]
        label = self.labels[idx]
        numFrames = self.numFrames[idx]
        inpSeq = []
        hogs = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrames, self.seqLen):
            #Frames of each video
            fl_name = vidName + '/' + 'frame' + str(int(round(i))) + '.jpg'
            img = Image.open(fl_name)
            #Extract Hog Features
            im = cv2.imread(fl_name, 0)
            h = hog.compute(im, winStride,padding,locations)
            # In case of augmenting horizontal flips, augment also the hog features again
            if self.spatial_transform(img.convert('RGB')).shape[0] == 2:
                hogs.append(torch.stack([torch.from_numpy(h), torch.from_numpy(h)], dim=0))
            else:
                hogs.append(torch.from_numpy(h))
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        hogs = torch.stack(hogs, 0)
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label, hogs

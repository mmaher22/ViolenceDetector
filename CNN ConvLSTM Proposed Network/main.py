import os
import sys
import glob
import torch
import argparse
from transformations import *
import numpy as np
import torch.nn as nn
from organizeData import *
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter

#Class for the Convolutional LSTM Gates
class ConvLSTM(nn.Module):
    def __init__(self, inputSZ, hiddenSZ, kernelSZ=3, stride=1, pad=1):
        super(ConvLSTM, self).__init__()
        self.inputSZ = inputSZ
        self.hiddenSZ = hiddenSZ
        self.Gates = nn.Conv2d(inputSZ + hiddenSZ, 4 * hiddenSZ, kernelSZ=kernelSZ, stride=stride, padding=pad)
        torch.nn.init.constant_(self.Gates.bias, 0)
        torch.nn.init.xavier_normal_(self.Gates.weight)

    def forward(self, input_, prevState):
        batchSZ = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prevState is None:
            stateSZ = [batchSZ, self.hiddenSZ] + list(spatial_size)
            prevState = (Variable(torch.zeros(stateSZ)), Variable(torch.zeros(stateSZ)))

        prevHidden, prev_cell = prevState
        stackedInput = torch.cat((input_, prevHidden), 1)
        #print('stacked inputs shape', stackedInput.shape)
        gates = self.Gates(stackedInput)
        inGate, remGate, outGate, cellGate = gates.chunk(4, 1)
        inGate = inGate.sigmoid()
        remGate = remGate.sigmoid()
        outGate = outGate.sigmoid()
        cellGate = cellGate.tanh()
        cell = (remGate * prev_cell) + (inGate * cellGate)
        hidden = outGate * cell.tanh()
        return hidden, cell

# Class for the CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, mem_size):
        super(CNNLSTMModel, self).__init__()
        self.mem_size = mem_size
        self.FF = nn.Linear(1764, 294) #feed forward network for reduction of HOG features
        #Pretrained resnet18 on ImageNET
        self.net = models.resnet18(pretrained = True)
        #self.alexnet = models.alexnet(pretrained=True)
        self.convNet = nn.Sequential(*list(self.net.children())[:-2])
        #self.convNet = nn.Sequential(*list(self.alexnet.features.children()))
        self.resNet = None
        #self.alexnet = None
        self.convLSTM = ConvLSTM(518, self.mem_size)
        #self.convLSTM = ConvLSTM(256, self.mem_size)
        self.RELU = nn.RELU()
        self.maxpool = nn.MaxPool2d(2)
        self.FF1 = nn.Linear(3*3*self.mem_size, 1000)
        self.batchNorm = nn.BatchNorm1d(1000)
        self.FF2 = nn.Linear(1000, 256)
        self.FF3 = nn.Linear(256, 10)
        self.FF4 = nn.Linear(10, 2)
        self.classifier = nn.Sequential(self.FF1, self.batchNorm, self.RELU, self.FF2, self.RELU, self.FF3, self.RELU, self.FF4)

    def forward(self, inputX, hogx):
        state = None
        seqLen = inputX.size(0) - 1
        for t in range(0, seqLen):
            #difference in hog features of the frames
            h1 = hogx[t] - hogx[t+1]
            h1 = self.FF(h1.view(-1, 1764))
            h1 = h1.view(-1, 6, 7, 7)
            #difference in video frames input
            inputX1 = inputX[t] - inputX[t+1]
            # pass through the pretrained networks
            inputX1 = self.convNet(inputX1)
            #concatenate the hog features with the output from the pretrained Convolutional network
            x2 = torch.cat([inputX1, h1], dim=1)
            state = self.convLSTM(x2, state)
        # max pooling of the output state of the ConvLSTM network
        inputX = self.maxpool(state[0])
        # classification of the output
        inputX = self.classifier(inputX.view(inputX.size(0), -1))
        return inputX

def makeSplit(violenceDir, noViolenceDir):
    #Folders for violent scenes
    imagesV = []
    for target in sorted(os.listdir(violenceDir)):
        d = os.path.join(violenceDir, target)
        imagesV.append(d)
    #Folders for non-violent scenes
    imagesNoV = []
    for target in sorted(os.listdir(noViolenceDir)):
        d = os.path.join(noViolenceDir, target)
        imagesNoV.append(d)
    # Whole dataset augmented together
    fullDataset = imagesV + imagesNoV
    # Labels of the dataset
    labels = list([1] * len(imagesV)) + list([0] * len(imagesNoV))
    # Total number of frames in the dataset
    numFrames = [len(glob.glob1(fullDataset[i], "*.jpg")) for i in range(len(fullDataset))]
    return fullDataset, labels, numFrames

def mainRun(outDir, violenceDirTrain, noviolenceDirTrain, violenceDirTest, noviolenceDirTest):
    numEpochs = 50 
    lr = 1e-4 
    stepSize = 25 
    decayRate = 0.5
    trainBatchSize = 16
    memSize = 256
    evalInterval = 5  
             
    trainDataset, trainLabels, trainNumFrames = makeSplit(violenceDirTrain, noviolenceDirTrain)
    testDataset, testLabels, testNumFrames = makeSplit(violenceDirTest, noviolenceDirTest)
    mean=[0.485, 0.456, 0.406] #mean of imagenet data
    std=[0.229, 0.224, 0.225] #standard deviation of imagenet data
    normalize = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #normalize like the imagenet data
    spatialTransform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224), ToTensor(), normalize])    
    vidSeqTrain = makeDataset(trainDataset, trainLabels, trainNumFrames, spatial_transform = spatialTransform)
    trainLoader = torch.utils.data.DataLoader(vidSeqTrain, batch_size=trainBatchSize,shuffle=True, num_workers=4, pin_memory=True)
    test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])
    vidSeqTest  = makeDataset(testDataset, testLabels, testNumFrames, spatial_transform = test_spatial_transform)
    testLoader = torch.utils.data.DataLoader(vidSeqTest, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)

    numTrainInstances = len(vidSeqTrain.images)
    numTestInstances = len(vidSeqTest.images)

    modelFolder = './resultsLogs_' + outDir # Dir for saving models and log files
    # Create the dir
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)
    # Log files
    writer = SummaryWriter(modelFolder)
    trainLogLoss = open((modelFolder + '/trainLoss.txt'), 'w')
    trainLogAcc = open((modelFolder + '/trainAcc.txt'), 'w')
    testLogLoss = open((modelFolder + '/testLoss.txt'), 'w')
    testLogAcc = open((modelFolder + '/testAcc.txt'), 'w')

    model = CNNLSTMModel(mem_size=memSize)

    trainParams = []
    for params in model.parameters():
        params.requires_grad = True
        trainParams += [params]
    model.train(True)

    lossFn = nn.CrossEntropyLoss()
    optimizerFn = torch.optim.RMSprop(trainParams, lr=lr)
    optimScheduler = torch.optim.lr_scheduler.StepLR(optimizerFn, stepSize, decayRate)

    minAcc = 0
    for epoch in range(numEpochs):
        optimScheduler.step()
        epochLoss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.train(True)
        print('Epoch = {}'.format(epoch + 1))
        writer.add_scalar('lr', optimizerFn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, targets, hog) in enumerate(trainLoader):
            iterPerEpoch += 1
            optimizerFn.zero_grad()
            inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4))
            hogVariable = Variable(hog.permute(1, 0, 2, 3))
            labelVariable = Variable(targets)
            #print('----->', inputVariable1.shape, hogVariable.shape)
            outputLabel = model(inputVariable1, hogVariable)
            loss = lossFn(outputLabel, labelVariable)
            loss.backward()
            optimizerFn.step()
            outputProb = torch.nn.Softmax(dim=1)(outputLabel)
            _, predicted = torch.max(outputProb.data, 1)
            #print('predicted: ', predicted, '\ntargets:', targets, ' --- ', (predicted == targets))
            numCorrTrain += (predicted == targets).sum()
            #print('Number of Correct: ', numCorrTrain, '---', numCorrTrain.double() / float(numTrainInstances))
            epochLoss += loss.item()
            #epochLoss += loss.data[0]
        avgLoss = epochLoss/iterPerEpoch
        trainAccuracy = (numCorrTrain.double() / float(numTrainInstances)) * 100
        print('Training: Loss = {} | Accuracy = {}% '.format(avgLoss, trainAccuracy))
        writer.add_scalar('train/epochLoss', avgLoss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        trainLogLoss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avgLoss))
        trainLogAcc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))

        if (epoch+1) % evalInterval == 0:
            model.train(False)
            print('Hold-Out Evaluation')
            testLossEpoch = 0
            testIter = 0
            numCorrTest = 0
            with torch.no_grad():
                for j, (inputs, targets, hog) in enumerate(testLoader):
                    testIter += 1
                    inputVariable1 = Variable(inputs[0])
                    hogVariable = Variable(hog[0])
                    labelVariable = Variable(targets)
                    outputLabel = model(inputVariable1, hogVariable)
                    outputLabel_mean = torch.mean(outputLabel, 0, True)
                    testLoss = lossFn(outputLabel_mean, labelVariable)
                    testLossEpoch += testLoss.item()
                    _, predicted = torch.max(outputLabel_mean.data, 1)
                    numCorrTest += (predicted == targets[0]).sum()
            testAcc = (numCorrTest.double() / float(numTestInstances)) * 100
            avgTestLoss = testLossEpoch / testIter
            print('Testing: Loss = {} | Accuracy = {}% '.format(avgTestLoss, testAcc))
            writer.add_scalar('test/epochloss', avgTestLoss, epoch + 1)
            writer.add_scalar('test/accuracy', testAcc, epoch + 1)
            testLogLoss.write('Test Loss after {} epochs = {}\n'.format(epoch + 1, avgTestLoss))
            testLogAcc.write('Test Accuracy after {} epochs = {}%\n'.format(epoch + 1, testAcc))
            if testAcc > minAcc:
                savePathClassifier = (modelFolder + '/bestModel.pth')
                torch.save(model, savePathClassifier)
                minAcc = testAcc
    trainLogAcc.close()
    testLogAcc.close()
    trainLogLoss.close()
    testLogLoss.close()
    writer.export_scalars_to_json(modelFolder + "/all_scalars.json")
    writer.close()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--violenceDirTrain', type=str)
    parser.add_argument('--noviolenceDirTrain', type=str)
    parser.add_argument('--violenceDirTest', type=str)
    parser.add_argument('--noviolenceDirTest', type=str)
    args = parser.parse_args()
    
    outDir = 'violenceOutput'
    violenceDirTrain = args.violenceDirTrain
    noviolenceDirTrain = args.noviolenceDirTrain
    violenceDirTest = args.violenceDirTest
    noviolenceDirTest = args.noviolenceDirTest
    
    mainRun(outDir, violenceDirTrain, noviolenceDirTrain, violenceDirTest, noviolenceDirTest)
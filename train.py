import torch
import torch.nn as nn
import numpy as np 
from model import GraphLSTMNet
from torch.autograd import Variable

def trainLSTM(trainLoader, numEpochs, classNum, featureSize) :
    lstm = GraphLSTMNet(featureSize, featureSize, featureSize, classNum)

    useGPU = torch.cuda.is_available()

    optimizer = torch.optim.SGD(lstm.parameters(), lr = 0.0001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(numEpochs):
        for index, data in enumerate(trainLoader) :
            torch.autograd.set_detect_anomaly(True)
            feature, label, neighbour, sequence, number = data
            label = label.long()
            if useGPU :
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                sequence = Variable(sequence.cuda())
                neighbour = Variable(neighbour.cuda())
                number = Variable(number.cuda())
            else :
                feature = Variable(feature).squeeze()
                label = Variable(label).squeeze()
                sequence = Variable(sequence).squeeze()
                neighbour = Variable(neighbour).squeeze()
                number = Variable(number).squeeze()

            outputs = lstm(feature, neighbour, number, sequence)
            loss = criterion(outputs, label)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return lstm, lstmloss

'''
Here to output pridiction of all nodes and then write another program to calculate accuracy
'''

'''
def testLSTM(model, testLoader) :
    useGPU = torch.cuda.is_available()

    #TODO: decide loss function here

    for epoch in range(numEpochs):
        for index, feature, label, neighbour, sequence, number in enumerate(testLoader) :
            if useGPU :
                index = Variable(index.cuda())
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                neighbour = Variable(neighbour.cuda())
                number = Variable(number.cuda())
            else :
                index = Variable(index)
                feature = Variable(feature())
                label = Variable(label())
                neighbour = Variable(neighbour())
                number = Variable(number())

            outputs = GraphLSTMNet(feature, neighbour, number, sequence)
            loss = criterion(outputs, label)
'''    
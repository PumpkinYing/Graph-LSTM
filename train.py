import torch
import numpy as np 
from model import GraphLSTMNet
from torch.autograd import Variable

def trainLSTM(trainLoader, numEpochs = 5, classNum) :
    index, feature, label, neighbour, sequence, number = enumerate(trainLoader)
    featureSize = feature.size(1)
    lstm = GraphLSTMNet(featureSize, featureSize, featureSize, classNum)

    useGPU = torch.cuda.is_available()

    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)

    #TODO: decide loss function here

    for epoch in range(numEpochs):
        for index, feature, label, neighbour, sequence, number in enumerate(trainLoader) :
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
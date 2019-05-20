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

    lstmloss = []

    for epoch in range(numEpochs):

        epochLoss = 0

        for index, data in enumerate(trainLoader) :
            torch.autograd.set_detect_anomaly(True)
            feature, label, neighbour, sequence, number = data
            label = label.long()

            feature = feature.squeeze()
            label = label.squeeze()
            neighbour = neighbour.squeeze()
            sequence = sequence.squeeze()
            number = number.squeeze()

            if useGPU :
                feature = feature.cuda()
                label = label.cuda()
                sequence = sequence.cuda()
                neighbour = neighbour.cuda()
                number = number.cuda()

            outputs = lstm(feature, neighbour, number, sequence)
            loss = criterion(outputs, label)
            epochLoss += loss.item()
            print(index, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(lstm.state_dict(),"checkpoint/CP{}.pth".format(epoch+1))
        print("Checkpoint {} Reached!".format(epoch+1))
        print("Epoch loss: {}".format(epochLoss/index))
        lstmloss.append(epochLoss/(index+1))

    print(lstmloss)

    return lstm, lstmloss

'''
Here to output pridiction of all nodes and then write another program to calculate accuracy
'''

def testLSTM(model, path, testPic) :
    useGPU = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    feature = torch.from_numpy(np.loadtxt(path+'/Feature/'+testPic)).squeeze(0)
    label = torch.from_numpy(np.loadtxt(path+'/Label/'+testPic)).squeeze(0)
    neighbour = torch.from_numpy(np.loadtxt(path+'/Neighbour/'+testPic)).squeeze(0)
    sequence = torch.from_numpy(np.loadtxt(path+'/Sequence/'+testPic)).squeeze(0)
    number = torch.from_numpy(np.loadtxt(path+'/Number/'+testPic)).squeeze(0)

    label = label.long()

    feature = feature.squeeze()
    label = label.squeeze()
    neighbour = neighbour.squeeze()
    sequence = sequence.squeeze()
    number = number.squeeze()

    if useGPU :
        feature = feature.cuda()
        label = label.cuda()
        sequence = sequence.cuda()
        neighbour = neighbour.cuda()
        number = number.cuda()

    outputs = model(feature, neighbour, number, sequence)
    return outputs
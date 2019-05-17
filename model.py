import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 
from torch.autograd import Variable

class GraphLSTMBlock(nn.Module) : 
    def __init__(self,inputSize,cellSize,hiddenSize) :
        super(GraphLSTMBlock,self).__init__()
        self.cellSize = cellSize
        self.hiddenSize = hiddenSize
        self.gate = nn.Linear(inputSize+hiddenSize,hiddenSize)
        self.selfGate = nn.Linear(inputSize,hiddenSize)
        self.neighbourGate = nn.Linear(hiddenSize,hiddenSize,bias = False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def step(self,input,hiddenState,cellState,index,neighbour,numNei) :
        combined = torch.cat((input, hiddenState[index]),1)
        neiHiddenState = torch.matmul(neighbour[i], hiddenSize) / numNei
        inputState = self.sigmoid(self.gate(combined) + self.neighbourGate(neiHiddenState))
        forgetSelf = self.selfGate(input)
        forgetSelfMat = torch.Tensor(hiddenState.size(0), hiddenState.size(1))
        forgetSelfMat = selfMat.copy_(forgetSelf)
        neiForgetState = self.sigmoid(forgetSelfMat + self.neighbourGate(hiddenState))
        forgetState = self.sigmoid(forgetSelf + self.neighbourGate(hiddenState[index]))
        outputState = self.sigmoid(self.gate(combined) + self.neighbourGate(neiHiddenState))
        helpCellState = self.tanh(self.gate(combined) + self.neighbourGate(neiHiddenState))
        curCellState = torch.matmul(neighbour,neiForgetState * cellState) / numNei + forgetState * cellState[index] + inputState * helpCellState
        curHiddenState = self.tanh(outputState * curCellState)

        return curCellState, curHiddenState

   def forward(self,inputs,neighbour,numNei,sequence) :
        cellState, hiddenState = initStates(inputs.size(0), inputs.size(1))
        residualHidden = hiddenState
        for index in sequence:
            curCellState, curHiddenState = self.step(inputs[index], hiddenState, cellState[index], index, neighbour[index], numNei[index])
            cellState[index] = curCellState
            hiddenState[index] = curHiddenState

        return hiddenState + residualHidden

    def initStates(size1,size2) :
        useGPU = torch.cuda.is_available()
        if(useGPU) :
            hiddenState = Variable(torch.randn(size1,size2).cuda())
            cellState = Variable(torch.randn(size1,size2).cuda())
        else :
            hiddenState = Variable(torch.randn(size1,size2))
            cellState = Variable(torch.randn(size1,size2))

        return cellState, hiddenState

class GraphLSTMNet(nn.Module) :
    def __init__(self, inputSize, cellSize, hiddenSize, classNum) :
        super(GraphLSTMNet,self).__init__()
        self.inputSize = inputSize
        self.cellSize = cellSize
        self.hiddenSize = hiddenSize
        self.classNum = classNum
        self.fc = nn.Linear(hiddenSize,classNum)
        self.GraphLSTMBlock1 = GraphLSTMBlock(inputSize, cellSize, hiddenSize)
        self.GraphLSTMBlock2 = GraphLSTMBlock(inputSize, cellSize, hiddenSize)

    def forward(self,inputs,neighbour,numNei,sequence) :
        out = self.GraphLSTMBlock1(inputs, neighbour, numNei, sequence)
        out = self.GraphLSTMBlock2(out, neighbour, numNei, sequence)
        out = self.fc(out)
        return out


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

    def step(self,inp,hiddenState,cellState,index,neighbour,numNei) :
        inp = inp.float()
        combined = torch.cat((inp, hiddenState[index]),0)
        neiHiddenState = torch.matmul(neighbour.double(), hiddenState.double()) / numNei
        neiHiddenState = neiHiddenState.float()
        inputState = self.sigmoid(self.gate(combined) + self.neighbourGate(neiHiddenState))
        forgetSelf = self.selfGate(inp.float())
        forgetSelfMat = torch.Tensor(hiddenState.size(0), hiddenState.size(1))
        forgetSelfMat = forgetSelfMat.copy_(forgetSelf)
        neiForgetState = self.sigmoid(forgetSelfMat + self.neighbourGate(hiddenState))
        forgetState = self.sigmoid(forgetSelf + self.neighbourGate(hiddenState[index]))
        outputState = self.sigmoid(self.gate(combined) + self.neighbourGate(neiHiddenState))
        helpCellState = self.tanh(self.gate(combined) + self.neighbourGate(neiHiddenState))
        curCellState = torch.matmul(neighbour.float(),neiForgetState.float() * cellState.float()) / numNei + forgetState.float() * cellState.float() + inputState.float() * helpCellState.float()
        curHiddenState = self.tanh(outputState * curCellState)

        return curCellState, curHiddenState

    def forward(self,inputs,neighbour,numNei,sequence) :
        cellState, hiddenState = self.initStates(inputs.shape[0], inputs.shape[1])
        residualHidden = hiddenState
        for index in sequence:
            index = int(index.item())
            curCellState, curHiddenState = self.step(inputs[index], hiddenState, cellState[index], index, neighbour[index], numNei[index])
            cellState[index] = curCellState
            hiddenState[index] = curHiddenState

        return hiddenState + residualHidden

    def initStates(self,size1,size2) :
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


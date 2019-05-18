import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 
from torch.autograd import Variable
import pynvml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
pynvml.nvmlInit()


class GraphLSTMBlock(nn.Module) : 
    def __init__(self,inputSize,cellSize,hiddenSize) :
        super(GraphLSTMBlock,self).__init__()
        self.cellSize = cellSize
        self.hiddenSize = hiddenSize
        self.gate = nn.Linear(inputSize+hiddenSize,hiddenSize).to(device)
        self.selfGate = nn.Linear(inputSize,hiddenSize).to(device)
        self.neighbourGate = nn.Linear(hiddenSize,hiddenSize,bias = False).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.tanh = nn.Tanh().to(device)

    def step(self,inp,hiddenState,cellState,index,neighbour,numNei) :
        inp = inp.float()
        combined = torch.cat((inp, hiddenState[index]),0)
        neiHiddenState = torch.matmul(neighbour.double(), hiddenState.double()) / numNei
        neiHiddenState = neiHiddenState.float()
        inputState = self.sigmoid(self.gate(combined) + self.neighbourGate(neiHiddenState))
        forgetSelf = self.selfGate(inp.float())
        forgetSelfMat = forgetSelf.expand(hiddenState.size(0), hiddenState.size(1))
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
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print(meminfo.used)
            mat_11 = torch.zeros((cellState.shape[0],cellState.shape[1])).to(device)
            mat_12 = torch.zeros((cellState.shape[0],cellState.shape[1])).to(device)
            mat_21 = torch.zeros((cellState.shape[0],cellState.shape[1])).to(device)
            mat_22 = torch.zeros((cellState.shape[0],cellState.shape[1])).to(device)
            mat_11[index] = curCellState
            mat_12[index] = cellState[index]
            mat_21[index] = curHiddenState
            mat_22[index] = hiddenState[index]
            cellState = cellState+mat_11-mat_12
            hiddenState = hiddenState+mat_21-mat_22

        return hiddenState + residualHidden

    def initStates(self,size1,size2) :
        # useGPU = torch.cuda.is_available()
        hiddenState = torch.randn(size1,size2).to(device)
        cellState = torch.randn(size1,size2).to(device)

        return cellState, hiddenState

class GraphLSTMNet(nn.Module) :
    def __init__(self, inputSize, cellSize, hiddenSize, classNum) :
        super(GraphLSTMNet,self).__init__()
        self.inputSize = inputSize
        self.cellSize = cellSize
        self.hiddenSize = hiddenSize
        self.classNum = classNum
        self.fc = nn.Linear(hiddenSize,classNum).to(device)
        self.GraphLSTMBlock1 = GraphLSTMBlock(inputSize, cellSize, hiddenSize)
        self.GraphLSTMBlock2 = GraphLSTMBlock(inputSize, cellSize, hiddenSize)

    def forward(self,inputs,neighbour,numNei,sequence) :
        out = self.GraphLSTMBlock1(inputs, neighbour, numNei, sequence)
        out = self.GraphLSTMBlock2(out, neighbour, numNei, sequence)
        out = self.fc(out)
        return out


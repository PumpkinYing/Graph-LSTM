import numpy as np 
import torch
import sys
import torch.nn as nn
from dataset import ParsingDataset
from torch.utils.data import DataLoader
from train import *

if __name__ == "__main__" :
    trainDataset = ParsingDataset("output")
    trainLoader = DataLoader(dataset = trainDataset, shuffle = True, pin_memory = torch.cuda.is_available())

    lstm, lstmLoss = trainLSTM(trainLoader, numEpochs = 5, classNum = 18, featureSize = 64)
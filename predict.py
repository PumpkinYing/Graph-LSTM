import numpy as np 
import torch
import sys
import os
import torch.nn as nn
from dataset import ParsingDataset
from torch.utils.data import DataLoader
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__' :

    lstm = GraphLSTMNet(64,64,64,18)
    '''
    if(device == "cuda") :
        lstm.cuda()
    else :
        lstm.cpu()
    '''
    lstm.load_state_dict(torch.load("checkpoint/CP5.pth"))

    fileNames = os.listdir("test/Feature/")
    for fn in fileNames :
        with torch.no_grad():
            output = testLSTM(lstm, "test", fn)
        predict = np.argmax(output.cpu().numpy(),axis = 1)
        np.savetxt("test/predict/"+fn,predict)


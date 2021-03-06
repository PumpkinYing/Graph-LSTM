import os
import torch
import torch.utils.data as data
import numpy as np

'''
    Read :
        Feature map of every node (input)
        Label of every node
        Neighbour of every node
        Updating sequence of each node
        Number of neighbouring node of each node
    File dir :
        train/test
        |---Feature
        |---Label
        |---Neighbour
        |---Sequence
        |---Number
'''
class ParsingDataset(data.Dataset) :
    def __init__(self, path) :
        super(ParsingDataset, self).__init__()
        self.nameList = os.listdir(path+'/Feature')
        self.size = len(self.nameList)
        self.path = path

    def __getitem__(self, index) :
        fileName = self.nameList[index]
        feature = torch.from_numpy(np.loadtxt(self.path+'/Feature/'+fileName)).squeeze(0)
        label = torch.from_numpy(np.loadtxt(self.path+'/Label/'+fileName)).squeeze(0)
        neighbour = torch.from_numpy(np.loadtxt(self.path+'/Neighbour/'+fileName)).squeeze(0)
        sequence = torch.from_numpy(np.loadtxt(self.path+'/Sequence/'+fileName)).squeeze(0)
        number = torch.from_numpy(np.loadtxt(self.path+'/Number/'+fileName)).squeeze(0)
        return [feature, label, neighbour, sequence, number]

    def __len__(self) :
        return self.size

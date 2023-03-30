import itertools
import math
import numpy as np
import pickle
import torch
from utils import * 

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class MyDataset(Dataset):
    def __init__(self,dataset,mode) -> None:
        super().__init__()
        assert mode in ["train","dev","test"]
        x_dir = "{}/data/{}.tensor".format(dataset,mode)
        y_dir = "{}/data/{}_y.tensor".format(dataset,mode)

        with open(x_dir,"rb") as f:
            self.x = torch.load(f)
        with open(y_dir,"rb") as f:
            self.y = torch.load(f) 

    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return (self.x[index],self.y[index])



    




dir = "stanford_tree_bank/data"
dts = MyDataset(dir,"train")



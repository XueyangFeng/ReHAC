import random

from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
from arguments import DataArguments
from transformers.data.data_collator import *
import datasets
from torch.utils.data import Dataset
import os

class TrainDataset(Dataset):
    def __init__(self, args: DataArguments):
        self.dataset = datasets.load_from_disk(dataset_path=args.train_data)['train']
        #self.args = args
        self.total_len = len(self.dataset)
        column_names = self.dataset.column_names
        print(f'Column names: {column_names}')
        num_rows = len(self.dataset)
        print(f'Number of rows: {num_rows}')

        features = self.dataset.features
        print(f'Column types: {features}')



        
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        trajectory = self.dataset[item]['trajectory']
        return self.dataset[item]['trajectory'], self.dataset[item]['em'], self.dataset[item]['f1']
        #return {key: torch.tensor(val) for key, val in self.dataset[item].items()}
        

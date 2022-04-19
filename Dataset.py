import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import spacy
import random
from itertools import permutations
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import util
from Vocabulary import Vocabulary




def build_traces(fileName, path, REBUILD_DATA):
    voc = Vocabulary(path)
    traces = voc.tokenizer(fileName, path, REBUILD_DATA)
    return traces, voc


def build_vocabulary(traces, voc):
    voc.build_vocabulary(traces)
    return voc



class Dataset(Dataset):
    def __init__(self, vocab, input, target, uncertain_subtraces = None):
        self.vocab = vocab
        self.input = self.vocab.numericalize(input)
        self.target = self.vocab.numericalize(target)
        self.uncertain_subtraces = uncertain_subtraces

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.uncertain_subtraces is None:
            return torch.tensor(input).to(device), torch.tensor(target).to(device)
        else:
            uncertain_subtraces = self.uncertain_subtraces[index]
            return torch.tensor(input).to(device), torch.tensor(target).to(device), uncertain_subtraces






class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        input = [item[0] for item in batch]
        input = pad_sequence(input, batch_first=True, padding_value=self.pad_idx)
        target =[item[1] for item in batch]
        target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)
        if len(batch[0]) == 2:
            return input, target
        else:
            uncertain_subtraces = [item[2] for item in batch]
            return input, target, uncertain_subtraces

def get_loader(
        vocab,
        input,
        target,
        batch_size = 512,
        num_workers=0,
        shuffle = True,
        pin_memory=False,
        uncertain_subtraces = None
):
    dataset = Dataset(vocab, input, target, uncertain_subtraces)
    pad_idx = dataset.vocab.act_to_index["<PAD>"]
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory=pin_memory,
        collate_fn = MyCollate(pad_idx=pad_idx)
    )
    return loader, dataset


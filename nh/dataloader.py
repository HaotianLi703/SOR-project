# -*- coding: utf-8 -*-
"""Dataloader for neural hawkes process.

Args:
    path: file path for the dataset
    batch_size: size of one batch

Results:
    A Dataset class
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CTLSTMDataset(Dataset):
    ''' Dataset class for neural hawkes data
    '''
    def __init__(self, file_path, ratio=1):
        self.file_path = file_path
        self.event_seqs = []
        self.time_seqs = []
        self.ratio = ratio

        with open(self.file_path, 'rb') as f:
            if 'dev' in file_path:
                seqs = pickle.load(f, encoding='latin1')['dev']
            elif 'train' in file_path:
                seqs = pickle.load(f, encoding='latin1')['train']
            elif 'test' in file_path:
                seqs = pickle.load(f, encoding='latin1')['test']
            seqs = random.sample(seqs, int(len(seqs) * self.ratio))
            for idx, seq in enumerate(seqs):
                # if idx == 1:
                #     print(seq[0].keys())
                self.event_seqs.append(torch.LongTensor([int(event['type_event']) for event in seq]))
                self.time_seqs.append(torch.FloatTensor([float(event['time_since_last_event']) for event in seq]))

    def __len__(self):
        return len(self.event_seqs)
    
    def __getitem__(self, index):
        sample = {
            'event_seq': self.event_seqs[index],
            'time_seq': self.time_seqs[index]
        }

        return sample

# def pad_batch_fn(batch_data):
#     sorted_batch = sorted(batch_data, key=lambda x: len(x['event_seq']), reverse=True)

#     event_seqs = [seq['event_seq'] for seq in sorted_batch]
#     time_seqs = [seq['time_seq'] for seq in sorted_batch]
#     seqs_length = list(map(len, event_seqs))

#     for idx, (event_seq, time_seq, seq_length) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
#         tmp_event_seq = torch.zeros(seqs_length[0])
#         tmp_event_seq[:seq_length] = torch.IntTensor(event_seq)
#         event_seqs[idx] = tmp_event_seq

#         tmp_time_seq = torch.zeros(seqs_length[0])
#         tmp_time_seq[:seq_length] = torch.FloatTensor(time_seq)
#         time_seqs[idx] = tmp_time_seq

#     return event_seqs, time_seqs, seqs_length

def pad_batch_fn(batch_data):
    sorted_batch = sorted(batch_data, key=lambda x: x['event_seq'].size(), reverse=True)
    event_seqs = [seq['event_seq'].long() for seq in sorted_batch]
    time_seqs = [seq['time_seq'].float() for seq in sorted_batch]
    seqs_length = torch.LongTensor(list(map(len, event_seqs))).to(device)
    last_time_seqs = torch.stack([torch.sum(time_seq) for time_seq in time_seqs]).to(device)

    event_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long().to(device)
    time_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float().to(device)

    for idx, (event_seq, time_seq, seqlen) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
        event_seqs_tensor[idx, :seqlen] = torch.LongTensor(event_seq).to(device)
        time_seqs_tensor[idx, :seqlen] = torch.FloatTensor(time_seq).to(device)
    
    return event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length

# def restore_batch(sample_batched, type_size):
#     event_seqs, time_seqs, seqs_length = sample_batched

#     event_seqs_list, time_seqs_list = [], []
#     total_time_list = []

#     for idx, (event_seq, time_seq, seq_length) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
#         tmp_event_seq = torch.ones(seq_length + 1, dtype=torch.int32) * type_size
#         tmp_event_seq[1:] = event_seq[:seq_length]
#         event_seqs_list.append(tmp_event_seq)

#         tmp_time_seq = torch.zeros(seq_length + 1, dtype=torch.float)
#         tmp_time_seq[1:] = time_seq[:seq_length]
#         time_seqs_list.append(tmp_time_seq)

#         total_time_list.append(torch.sum(tmp_time_seq))
    
#     return event_seqs_list, time_seqs_list, total_time_list

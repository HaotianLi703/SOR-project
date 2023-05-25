import pickle
import time
import datetime
import torch
from torch.utils.data import DataLoader

import dataloader
import CTLSTM
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(settings):
    '''Testing process'''
    model_name = settings['model_name']
    test_path = settings['test_path']
    batch_size = settings['batch_size']

    test_loss = 0.0
    test_event_num = 0.0

    with open('nh/' + model_name, 'rb') as f:
        model = pickle.load(f)
    
    model = model.to(device)
    test_dataset = dataloader.CTLSTMDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=dataloader.pad_batch_fn, shuffle=True)

    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs,seqs_length)
            
            test_event_num += torch.sum(seqs_length)
            test_loss -= likelihood

        print('Test set\nTest Likelihood per event: {:5f} nats\n'.format(-test_loss/test_event_num))

if __name__ == "__main__":
    settings = {
        'model_name': 'model_125_10_320',
        'test_path':'nh/data_retweet/test.pkl',
        'batch_size': 256
    }

    test(settings)
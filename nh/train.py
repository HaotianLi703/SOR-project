# -*- coding: utf-8 -*-
"""Training code for neural hawkes model."""
# from copyreg import pickle
from lzma import MODE_NORMAL
import pickle
import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

import dataloader
import CTLSTM
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(settings):
    """Training process."""
    hidden_size = settings['hidden_size']
    type_size = settings['type_size']
    train_path = settings['train_path']
    dev_path = settings['dev_path']
    batch_size = settings['batch_size']
    epoch_num = settings['epoch_num']
    current_date = settings['current_date']
    ratio = settings['ratio']
    pre_model = settings['pre_model']

    if pre_model:
        with open(pre_model, 'rb') as f:
            model = pickle.load(f)
        model = model.to(device)
    else:
        model = CTLSTM.CTLSTM(hidden_size, type_size).to(device)
    optim = opt.Adam(model.parameters())
    train_dataset = dataloader.CTLSTMDataset(train_path, ratio)
    dev_dataset = dataloader.CTLSTMDataset(dev_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dataloader.pad_batch_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=256,collate_fn=dataloader.pad_batch_fn, shuffle=True)

    last_dev_loss = 0.0
    for epoch in range(epoch_num):
        model.train()
        tic_epoch = time.time()
        epoch_train_loss = 0.0
        epoch_dev_loss = 0.0
        train_event_num = 0
        dev_event_num = 0
        print('Epoch.{} starts.'.format(epoch))
        tic_train = time.time()
        for i_batch, sample_batched in enumerate(train_dataloader):
            tic_batch = time.time()
            
            optim.zero_grad()
            
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)
            batch_event_num = torch.sum(seqs_length)
            batch_loss = -likelihood

            batch_loss.backward()
            optim.step()
            
            toc_batch = time.time()
            if i_batch % 10 == 0:
                print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'.format(epoch, i_batch, likelihood/batch_event_num, toc_batch-tic_batch))
            epoch_train_loss += batch_loss
            train_event_num += batch_event_num

        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Likelihood per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, -epoch_train_loss/train_event_num, toc_train-tic_train))

    tic_eval = time.time()
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dev_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs,seqs_length)
            
            dev_event_num += torch.sum(seqs_length)
            epoch_dev_loss -= likelihood

        toc_eval = time.time()
        toc_epoch = time.time()
        print('Epoch.{} Devlopment set\nDev Likelihood per event: {:5f} nats\nEval Time:{:2f}s.\n'.format(epoch, -epoch_dev_loss/dev_event_num, toc_eval-tic_eval))
    
        with open("nh/loss_{}.txt".format(current_date), 'a') as l:
            l.write("Epoch {}:\n".format(epoch))
            l.write("Train Event Number:\t\t{}\n".format(train_event_num))
            l.write("Train Likelihood per event:\t{:.5f}\n".format(-epoch_train_loss/train_event_num))
            l.write("Training time:\t\t\t{:.2f} s\n".format(toc_train-tic_train))
            l.write("Dev Event Number:\t\t{}\n".format(dev_event_num))
            l.write("Dev Likelihood per event:\t{:.5f}\n".format(-epoch_dev_loss/dev_event_num))
            l.write("Dev evaluating time:\t\t{:.2f} s\n".format(toc_eval-tic_eval))
            l.write("Epoch time:\t\t\t{:.2f} s\n".format(toc_epoch-tic_epoch))
            l.write("\n")
    
    with open(f"nh/model_{int(20000 * ratio)}_{batch_size}_{epoch_num + int(pre_model.split('_')[-1]) if pre_model else 0}", 'wb') as f:
        pickle.dump(model, f)
        
    # gap = epoch_dev_loss/dev_event_num - last_dev_loss
    # if abs(gap) < 0.1:
    #     print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
    #     break
    
    # last_dev_loss = epoch_dev_loss/dev_event_num
    
    return


if __name__ == "__main__":
    settings = {
        'hidden_size': 32,
        'type_size': 3,
        'train_path': 'nh/data_retweet/train.pkl',
        'dev_path': 'nh/data_retweet/dev.pkl',
        'batch_size': 10,
        'epoch_num': 10,
        'ratio': 0.00625,
        'current_date': datetime.date.today(),
        'pre_model' : 'nh/model_125_10_270'
    }

    train(settings)

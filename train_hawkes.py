import data_processers as dp
import numpy 
import torch    
from torch.backends import cudnn 

def train_hawkes():

    # 读取和预处理数据
    print('Reading and processing data for training vanilla hawkes process...')
    data_process = dp.DataProcesser(
        {
        'path_rawdata': 'data/data_retweet/', 
            'size_batch': 10, 
            'ratio_train':numpy.float32(1.0),
            'to_read': ['train', 'dev'],
            'partial_predict': 0
        }
    )

    # compile the model

    # define the loss function

    # define the optimizer

    # train the model

    for epoch in range(100):
        print('Epoch %d' % epoch)
        pass

if __name__ == '__main__':
    train_hawkes()
import data_processers as dp
import numpy 
import torch    
from torch.backends import cudnn 
import model_hawkes

def train_hawkes():

    # 读取和预处理数据
    print('Reading and processing data for training inhibit hawkes process...')
    data_process = dp.DataProcesser(
        {        'path_rawdata': 'data/data_retweet/', 
            'size_batch': 10, 
            'ratio_train':numpy.float32(1.0),
            'to_read': ['train', 'dev'],
            'partial_predict': 0
        }
    )

    # compile the model
    model = model_hawkes.Hawks_Inhibit(type_num=3)
    # define the loss function

    # define the optimizer

    # train the model
    max_step = 1
    epo = 1
    for epoch in range(epo):
        print('Epoch %d' % epoch)

        for step_train in range(max_step):
            data_process.process_data(
                tag_batch = 'train',
                idx_batch_current = step_train,
                tag_model = 'hawkesinhib',
                multiple = numpy.int32(1),
                predict_first = 1
            )
            input_list = [data_process.seq_time_to_current_numpy,
                data_process.seq_type_event_numpy,
                data_process.time_since_start_to_end_numpy,
                data_process.num_sims_start_to_end_numpy,
                data_process.seq_mask_numpy,
                data_process.seq_mask_to_current_numpy,
                data_process.seq_sims_time_to_current_numpy,
                data_process.seq_sims_mask_to_current_numpy,
                data_process.seq_sims_mask_numpy
            ]

            out = model.forward(input_list)

            print(1)

if __name__ == '__main__':
    train_hawkes()
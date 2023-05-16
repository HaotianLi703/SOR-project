import data_processers
import numpy
import torch

def test_hawkes_inhibit():
    # 在测试集上测试模型
    # 读取和预处理数据
    print('Reading and processing data for testing inhibit hawkes process...')
    data_process = data_processers.DataProcesser(
        {
            'path_rawdata': 'data/data_retweet/',
            'size_batch': 1,
            'ratio_train': numpy.float32(0.0),
            # test和test1什么区别？pending check
            'to_read': ['test'],
            'partial_predict': 0
        }
    )

    # 读入训练好的模型
    model = torch.load('model_hawkesinhib.pkl')

    # model传输到GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 记录测试集上的总log-likelihood
    total_log_likelihood = 0
    # 记录测试集的总事件数
    total_num_events = 0

    # 在测试集上测试模型
    for step_test in range(data_process.max_nums['test']):
        data_process.process_data(
            tag_batch = 'test',
            idx_batch_current = step_test,
            tag_model = 'hawkesinhib',
            # multiple和predict_first的值是什么意思？pending check
            multiple = numpy.int32(10),
            predict_first = 1
        )
        input_list_test = [data_process.seq_time_to_current_numpy,
            data_process.seq_type_event_numpy,
            data_process.time_since_start_to_end_numpy,
            data_process.num_sims_start_to_end_numpy,
            data_process.seq_mask_numpy,
            data_process.seq_mask_to_current_numpy,
            data_process.seq_sims_time_to_current_numpy,
            data_process.seq_sims_mask_to_current_numpy,
            data_process.seq_sims_mask_numpy
        ]

        # 转换为tensor
        for index in range(len(input_list_test)):
            input_list_test[index] = torch.from_numpy(input_list_test[index])
        
        # 传输到GPU
        if torch.cuda.is_available():
            for i in range(len(input_list_test)):
                input_list_test[i] = input_list_test[i].cuda()
        
        out = model.forward(input_list_test)

        # 记录测试集上的总log-likelihood
        total_log_likelihood += out[1].item()
        # 记录测试集的总事件数
        total_num_events += out[-1].item()

    # 计算测试集上的平均log-likelihood
    avg_log_likelihood = round(total_log_likelihood / total_num_events, 4)

    print('Average log-likelihood on test set: %f' % avg_log_likelihood)


if __name__ == '__main__':
    test_hawkes_inhibit()
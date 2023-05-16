import data_processers as dp
import numpy 
import torch    
from torch.backends import cudnn 
import model_hawkes


def train_hawkes():
    # 训练记录日志
    my_log = dict()
    # 原文中对应track_period，默认1000，为每隔若干个个batch做一次validation
    my_log['track_per_batch'] = 1000
    # 记录当前batch的数量
    my_log['iteration'] = 0
    # 记录当前正在进行的validation的信息
    my_log['val_tracking'] = dict()
    my_log['val_tracking']['val_log_lik'] = []
    # 记录validation的最佳结果
    my_log['best_log_lik_val'] = -100000

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

    # model传输到GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # train the model
    epo = 50
    for epoch in range(epo):
        print('Epoch %d' % epoch)

        # 每个epoch前打乱数据
        data_process.shuffle_train_data()

        for step_train in range(data_process.max_nums['train']):
            # multiple和predict_first的值是什么意思？pending check
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

            # 转换为tensor
            for index in range(len(input_list)):
                input_list[index] = torch.from_numpy(input_list[index])

            # 传输到GPU
            if torch.cuda.is_available():
                for i in range(len(input_list)):
                    input_list[i] = input_list[i].cuda()

            out = model.forward(input_list)

            # 计算loss, 即-log-likelihood
            loss = my_loss_function(out[0])

            # 清空梯度
            optimizer.zero_grad()


            # 反向传播
            loss.backward()

            # print('loss为  ' + str(loss.item()))

            # 更新参数
            optimizer.step()

            # 限制delta的范围大于0
            model.delta.data = torch.clamp(model.delta.data, min=0.0)

            # 记录当前batch的数量
            my_log['iteration'] += 1

            if my_log.get('iteration') % my_log.get('track_per_batch')==0:
                # validation
                print('Validating,....')
                # 记录所有的log-likelihood的加和
                total_loglik_val = 0
                # 记录validation中事件的数量;``
                total_event_val = 0

                for val_step in range(data_process.max_nums['dev']):
                    # multiple和predict_first的值是什么意思？pending check
                    data_process.process_data(
                            tag_batch = 'dev    ',
                            idx_batch_current = val_step,
                            tag_model = 'hawkesinhib',
                            multiple = numpy.int32(10),
                            predict_first = 1
                        )
                    input_list_val = [data_process.seq_time_to_current_numpy,
                    data_process.seq_type_event_numpy,
                    data_process.time_since_start_to_end_numpy,
                    data_process.num_sims_start_to_end_numpy,
                    data_process.seq_mask_numpy,
                    data_process.seq_mask_to_current_numpy,
                    data_process.seq_sims_time_to_current_numpy,
                    data_process.seq_sims_mask_to_current_numpy,
                    data_process.seq_sims_mask_numpy
                    ] 
                    for index in range(len(input_list_val)):
                        input_list_val[index] = torch.from_numpy(input_list_val[index])

                    # 传输到GPU
                    if torch.cuda.is_available():
                        # model = model.cuda()
                        for i in range(len(input_list_val)):
                            input_list_val[i] = input_list_val[i].cuda()

                    out_val = model.forward(input_list_val)
                    assert out_val[1].item() < 0, 'log-likelihood 不应该是正的!'
                    # 计算loss, 即-log-likelihood
                    loss_val = my_loss_function(out_val[0])

                    # 记录所有的log-likelihood的加和
                    total_loglik_val += out_val[1].item()
                    # 记录validation中事件的数量
                    total_event_val += out_val[-1].item()
                    assert total_event_val > 0, '事件的数量不应该小于0'
                    
                    if val_step % 50 == 0:
                        print('已经验证了' + str(val_step) + '个batch，总共有' + str(data_process.max_nums['dev']) + '个batch')
                
                # 记录当前validation的平均log-likelihood
                my_log['val_tracking']['val_log_lik'].append(round(total_loglik_val / total_event_val, 4))

                # 如果当前的平均log-likelihood比之前的最好的还要好，就更新最好的结果，并且保存模型
                if my_log['val_tracking']['val_log_lik'][-1] > my_log['best_log_lik_val']:
                    print('发现了更好的模型，其log-likelihood为'+str(my_log['val_tracking']['val_log_lik'][-1]) +'之前最好的为'+ str(my_log['best_log_lik_val']))
                    my_log['best_log_lik_val'] = my_log['val_tracking']['val_log_lik'][-1]
                    torch.save(model, 'model_hawkesinhib.pkl')
                    
                    print('保存了模型')



    
def my_loss_function(x):
    """
    自定义损失函数，传入的是-log-likelihood
    """
    return x

if __name__ == '__main__':
    train_hawkes()
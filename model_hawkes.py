import torch
import numpy
import os

class Hawkes(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

class Hawks_Inhibit:
    def __init__(self, type_num):
        # 定义参数
        self.mu = torch.rand((type_num, ), requires_grad=True)
        self.alpha = torch.rand((type_num, type_num), requires_grad=True)
        self.delta = torch.rand((type_num, type_num), requires_grad=True)
        self.__parameters = dict(mu=self.mu, alpha=self.alpha, delta=self.delta)
        self.___gpu = False

    def forward(self, input_list):
        '''
        前向传播，输出log-likelihood
        use this function to compute negative log likelihood
        seq_time_to_end : T * size_batch -- T-t_i
        seq_time_to_current : T * T * size_batch --
        for each batch, it is T * T, and at each time step t,
        it tracks the ( t_i - t_i' ) for all t_i' < t_i
        seq_type_event : T * size_batch -- for each data
        and each time step, tracks the type of event k_i
        time_since_start_to_end : size_batch -- time for seq
        num_sims_start_to_end : size_batch -- N for each seq
        #
        seq_mask : T * size_batch -- 1/0
        seq_mask_to_current : T * T * size_batch -- 1/0
        #
        seq_sims_mask : N * size_batch -- 1/0
        '''

        for index in range(len(input_list)):
            input_list[index] = torch.from_numpy(input_list[index])

        [seq_time_to_current_numpy, 
        seq_type_event_numpy, 
        time_since_start_to_end_numpy,
        num_sims_start_to_end_numpy,
        seq_mask_numpy,
        seq_mask_to_current_numpy,
        seq_sims_time_to_current_numpy,
        seq_sims_mask_to_current_numpy,
        seq_sims_mask_numpy] = input_list
        
        alpha_over_seq = self.alpha[:, seq_type_event_numpy.detach().numpy()]
        delta_over_seq = self.delta[:, seq_type_event_numpy.detach().numpy()]

        lambda_over_seq_sims_tilde = self.mu[:,None,None] + torch.sum(
            (
                seq_sims_mask_to_current_numpy[None,:,:,:] * (
                    alpha_over_seq[:,None,:,:] * torch.exp(
                        -delta_over_seq[:,None,:,:] * seq_sims_time_to_current_numpy[None,:,:,:]
                    )
                )
            ), axis=2
        ) # dim_process * N * size_batch

        # soft relu
        lambda_over_seq_sims = torch.log(
            numpy.float32(1.0) + torch.exp(
                lambda_over_seq_sims_tilde
            )
        )
        # dim_process * N * size_batch      

        # dim_process * N * size_batch
        #
        lambda_sum_over_seq_sims = torch.sum(
            lambda_over_seq_sims, axis=0
        )
        # N * size_batch
        # mask the lambda of simulations
        lambda_sum_over_seq_sims *= seq_sims_mask_numpy
        #
        # 计算积分项的无偏估计
        term_3 = torch.sum(
            lambda_sum_over_seq_sims, axis=0
        ) * time_since_start_to_end_numpy / num_sims_start_to_end_numpy
        # (size_batch, )
        term_2 = numpy.float32(0.0)
        #
        '''
        for this model, the computation of term_3 follows the same procedure of term_1, since we need to estimate lambda(s_j), i.e, we need large N * T * size_batch tensors for (1) time to current; (2) mask for (1).
        then we can just follow the steps of term_1 to finish the integral estimation.
        correspondingly, we need to modify the data processors, to generate the big tensors
        '''
        # then we compute the 1st term, which is the trickest
        # we use seq_time_to_current : T * T * size_batch
        # seq_mask_to_current : T * T * size_batch
        lambda_over_seq_tilde = self.mu[:, None, None] + torch.sum(
            (
                seq_mask_to_current_numpy[None,:,:,:]
                * (
                    alpha_over_seq[:,None,:,:] * torch.exp(
                        -delta_over_seq[:,None,:,:]
                        * seq_time_to_current_numpy[None,:,:,:]
                    )
                )
            )
            , axis=2
        ) # dim_process * T * size_batch
        #
        lambda_over_seq = torch.log(
            numpy.float32(1.0) + torch.exp(
                lambda_over_seq_tilde
            )
        ) # dim_process * T * size_batch
        #
        lambda_sum_over_seq = torch.sum(
            lambda_over_seq, axis=0
        ) # T * size_batch
        # now we choose the right lambda for each step
        # by using seq_type_event : T * size_batch
        new_shape_0 = lambda_over_seq.shape[1]*lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        #
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]
        #
        lambda_target_over_seq = lambda_over_seq.permute(
            (1,2,0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            torch.arange(new_shape_0),
            seq_type_event_numpy.flatten().detach().numpy()
        ].reshape(
            (back_shape_0, back_shape_1)
        )
        # T * size_batch
        # if there is NaN,
        # it can also be the issue of underflow here
        log_lambda_target_over_seq = torch.log(
            lambda_target_over_seq + numpy.float32(1e-9)
        )
        log_lambda_target_over_seq *= seq_mask_numpy
        #
        log_lambda_sum_over_seq = torch.log(
            lambda_sum_over_seq + numpy.float32(1e-9)
        )
        log_lambda_sum_over_seq *= seq_mask_numpy
        #
        term_1 = torch.sum(
            log_lambda_target_over_seq, axis=0
        )
        term_sum = torch.sum(
            log_lambda_sum_over_seq, axis=0
        )
        # (size_batch, )
        #
        '''
        log-likelihood computed in this section is batch-wise
        '''
        log_likelihood_seq_batch = torch.sum(
            term_1 - term_2 - term_3
        )
        log_likelihood_type_batch = torch.sum(
            term_1 - term_sum
        )
        log_likelihood_time_batch = log_likelihood_seq_batch - log_likelihood_type_batch
        #
        # cost_to_optimize = -log_likelihood_seq_batch + self.term_reg
        cost_to_optimize = -log_likelihood_seq_batch
        #
        log_likelihood_seq = log_likelihood_seq_batch
        log_likelihood_type = log_likelihood_type_batch
        log_likelihood_time = log_likelihood_time_batch


        return cost_to_optimize, log_likelihood_seq, log_likelihood_type, log_likelihood_time
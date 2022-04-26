import torch
import random
import numpy as np
import time
import sys
import itertools


def statistic_old_new(pred, truth, new_rel):
    too, tnn, foo, fnn, fon, fno = 0, 0, 0, 0, 0, 0
    for p, t in zip(pred.reshape(-1).tolist(), truth.reshape(-1).tolist()):
        if p == t:
            if t in new_rel:
                tnn += 1
            else:
                too += 1
        else:
            if t in new_rel and p in new_rel:
                fnn += 1
            elif t not in new_rel and p in new_rel:
                fon += 1
            elif t in new_rel and p not in new_rel:
                fno += 1
            else:
                foo += 1
    # return (oo,nn,on,no),(oo/(oo+on),nn/(no+nn))#(旧关系预测对数目，旧关系预测成新关系数目，新关系预测成旧关系数目，新关系预测对数目），（两个recall，对应的是TPR和1-FRP）
    return np.array((too, tnn, foo, fnn, fon, fno))


def batch2device(batch_tuple, device):
    ans = []
    for var in batch_tuple:
        if isinstance(var, torch.Tensor):
            ans.append(var.to(device))
        elif isinstance(var, list):
            ans.append(batch2device(var, device))
        elif isinstance(var, tuple):
            ans.append(tuple(batch2device(var, device)))
        else:
            ans.append(var)
    return ans


def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)


class outputer(object):

    def __init__(self):
        self.start_time = time.time()
        self.all_results = []
        self.result_all_test_data = []
        self.sequence_times = 0

    def init(self):
        self.start_time = time.time()
        self.all_results = []
        self.result_all_test_data = []
        self.sequence_times = 0

    def append(self, sequence_results=None, result_whole_test=None):
        if not isinstance(sequence_results, type(None)):
            self.all_results.append(sequence_results)
        if not isinstance(result_whole_test, type(None)):
            self.result_all_test_data.append(result_whole_test)
        self.sequence_times += 1.0

    def print_avg_results(self, all_results):
        avg_result = []
        for i in range(len(all_results[0])):  # 代表了k shot
            avg_result.append(np.average([result[i] for result in all_results], 0))
        for line_result in avg_result:
            self.print_list(line_result)
        return avg_result

    def print_avg_cand(self, sample_list):
        cand_lengths = []
        for sample in sample_list:
            cand_lengths.append(len(sample[1]))
        print('avg cand size:', np.average(cand_lengths))

    def print_list(self, result):
        for num in result:
            sys.stdout.write('%.3f, ' % num)
        print('')

    def output(self):
        avg_result_all_test = np.average(self.result_all_test_data, 0)
        for result_whole_test in self.result_all_test_data:
            self.print_list(result_whole_test)
        print("-------------------------------------------")
        self.print_list(avg_result_all_test)
        print("===========================================")
        all_results = self.print_avg_results(self.all_results)
        # np.array(self.all_results)
        # print (all_results)
        print(all_results[-1].mean())

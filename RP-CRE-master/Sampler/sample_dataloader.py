# -- coding: utf-8 --
import collections
import torch
import numpy as np
import json
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset

class sample_dataloader(object):

    def __init__(self, quadruple, memory, id2sent, config=None, seed=None):
        self.quadruple = quadruple
        self.id2sent = id2sent
        self.data_idx = self.get_contrastive_data(memory, config.batch_size)
        self.config = config
        self._ix = 0

        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

    def __len__(self):
        return len(self.data_idx)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            l_idx, is_p_data = self.data_idx[self._ix]
        except IndexError:
            # Possibly reset `self._ix`?
            self._ix = 0
            raise StopIteration

        batch_all = [self.quadruple[i] for i in l_idx]
        sent_inp, emb_inp, preds, trues = [], [], [], []
        for (sent_id, emb, pred, true) in batch_all:
            sent_inp.append(torch.tensor(self.id2sent[sent_id]).int())
            emb_inp.append(emb)
            preds.append(pred)
            trues.append(true)
        labels = np.zeros(shape=(len(trues), len(trues)))
        if is_p_data:
            comparison = torch.ones((len(trues), len(trues)))
            for i in range(1, len(trues)):
                for j in range(0, i):
                    labels[i][j] = (trues[i] == trues[j])
        else:
            comparison = torch.ones((len(trues), len(trues)))
            for i in range(1, len(trues)):
                for j in range(0, i):
                    labels[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) * (trues[i] == trues[j])
                    # AB+AB^C+A^BC
                    comparison[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) or (
                            (preds[i] == trues[i]) * (preds[j] != trues[j]) + (preds[i] != trues[i]) * (
                            preds[j] == trues[j])) * (trues[i] == trues[j])
        labels += labels.T
        for i in range(len(trues)):
            labels[i][i] = (preds[i] == trues[i])
        labels = torch.IntTensor(labels)

        self.verify_metrix(labels, comparison)
        sent_inp = torch.stack(sent_inp)
        emb_inp = torch.stack(emb_inp)
        self._ix += 1
        return sent_inp, emb_inp, labels, comparison

    def verify_metrix(self, labels, comparison):
        """
        labels和comparison是两个相同的方阵，设计实现以下功能：
        检测以下情况，不满足时候抛出错误：
        1. labels为1时comparison为True(1)
        2. comparsion为0时labels为0
        要求尽量使用矩阵操作
        """
        if comparison[labels == 1].sum() < comparison[labels == 1].shape[0] or labels[comparison == 0].sum() != 0:
            raise Exception('labels and comparison not matched')

    def get_contrastive_data(self, memory, batch_size):
        """
        From
        1. Positive and negative embeddings in one sentence.
        2. Between cluster
        3. In cluster different sentence positive embedding.
        4. From memory.
        """
        # 根据句子簇分类,得到分好类的变量假设是 cluster2sentences

        # import pandas as pd
        # col_name=["sent_id", "embedding", "pred_id", "true_id"]
        # df=pd.DataFrame(zip(*quadruple),columns=col_name)
        # df["is_p"]=(df["pred_id"]==df["true_id"])
        # df["is_n"] = (df["pred_id"] != df["true_id"])
        # # 1. Positive and negative embeddings in one sentence
        # stat=df.groupby("sent_id")[["is_p"]].count().reset_index().describe()
        # stat_n = df.groupby("sent_id")[["is_n"]].sum().reset_index().describe()
        # print(f"Median of same sentence sample:{stat.loc['50%']}, min of same sentence sample:{stat.loc['min']}\nMedian of negative sample:{stat_n.loc['50%']}, min of same sentence sample:{stat_n.loc['min']}")
        #
        # p_pool = df.loc[df["is_p"] == True].values.tolist()
        #
        # # get positive negetive pool
        # pt_pool = sorted(quadruple, key=lambda x: x[0]) if batch_size < median_num else None
        # sample by batch

        # if pt_pool is not None:
        pn_batch_idx = []
        p_idx = []
        p_batch_idx = []  # get positive pool
        sent_id2ins_id = collections.defaultdict(list)
        count = 0

        for all_items in memory.values():
            for _, quad_ins in all_items:
                self.quadruple.append(quad_ins)

        for i, quad in enumerate(self.quadruple):
            sent_id2ins_id[quad[0]].append(i)
            if quad[-1] == quad[-2]:
                p_idx.append(i)

        p_batch_idx = [(i, 1) for i in self.multi_sample_no_replace(p_idx, batch_size)]
        # negative
        for ins in sent_id2ins_id.values():
            n = len(ins) / batch_size
            if n >= 1:
                for i in self.multi_sample_no_replace(ins, batch_size):
                    pn_batch_idx.append((i, 0))
                # for i in range(round(n - 0.1)):
                #     pn_batch_idx.extend(np.random.choice(a=ins, size=batch_size, replace=False).tolist())
            else:
                if n > 0.8:  # just sample when enough positive and negative samples
                    count += 1
                    print(f"Over sampling num:{count}")
                    pn_batch_idx.append((np.random.choice(a=ins, size=batch_size, replace=True).tolist(), 0))
        p_batch_idx.extend(pn_batch_idx)

        random.shuffle(p_batch_idx)
        return p_batch_idx

    def multi_sample_no_replace(self, list_collection, n, shuffle=True):
        if shuffle:
            random.shuffle(list_collection)
        return list(self.split_list_by_n(list_collection, n, last_to_n=True))

    def split_list_by_n(self, list_collection, n, last_to_n=False):
        """
        将list均分，每份n个元素
        :return:返回的结果为评分后的每份可迭代对象
        """

        for i in range(0, len(list_collection), n):
            if last_to_n:
                if (i + n) > len(list_collection):
                    yield list_collection[i:] + np.random.choice(a=list_collection, size=i + n - len(list_collection),
                                                                 replace=False).tolist()
                else:
                    yield list_collection[i: i + n]

            else:
                yield list_collection[i: i + n]

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class memory_fn(object):

    def __init__(self, id2sent):
        self.id2sent = id2sent

    def collect_fn(self, batch_data):
        labels = []
        tokens_id = []
        tokens = []
        for ins in batch_data:
            labels.append(torch.tensor(ins[-1]))
            tokens_id.append(torch.tensor(ins[0]))
            tokens.append(torch.tensor(self.id2sent[ins[0]]))
        labels = torch.stack(labels, dim=0)
        tokens = torch.stack(tokens, dim=0)
        tokens_id = torch.stack(tokens_id, dim=0)
        return (labels, tokens, tokens_id)


class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path,
                                                       additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task  # 每一轮任务进入几个新关系

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            self.batch == 0
            raise StopIteration()

        indexs = self.shuffle_index[self.config.rel_per_task * self.batch: self.config.rel_per_task * (self.batch + 1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for i in range(self.config.num_of_relation)]
        val_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]
        self.id2sent = {}
        for j, relation in enumerate(data.keys()):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample["tokens_id"] = j * len(rel_samples) + i
                tokenized_sample['relation'] = self.rel2id[sample['relation']]
                tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.config.max_length)
                self.id2sent[j * len(rel_samples) + i] = tokenized_sample['tokens']
                if self.config.task_name == 'FewRel':
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:  # 一个关系最多320个样本
                            break
        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def get_id2sent(self):
        return self.id2sent

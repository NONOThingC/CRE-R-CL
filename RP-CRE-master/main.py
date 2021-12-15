import collections
import itertools
import pickle
from math import ceil

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from config import Config

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified

from utils import outputer

from sampler import data_sampler
from data_loader import get_data_loader

def transfer_to_device(list_ins, device):
    import torch
    for ele in list_ins:
        if isinstance(ele,list):
            for x in ele:
                x.to(device)
        if isinstance(ele,torch.Tensor):
            ele.to(device)
    return list_ins

# Done
def get_proto(config, encoder, mem_set):
    # aggregate the prototype set for further use.
    data_loader = get_data_loader(config, mem_set, False, False, 1)
  
    features = []
    for step, (labels, tokens) in enumerate(data_loader):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        features.append(feature)
    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True)
    
    # return the averaged prototype
    return proto

# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(config, encoder, sample_set):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []

    for step, (labels, tokens) in enumerate(data_loader):
        tokens=torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = encoder(tokens).cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        mem_set.append(instance)
    return mem_set

def train_simple_model(config, encoder, classifier, training_data, epochs):

    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
                            {'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}
                            ])

    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens) in enumerate(data_loader):
            encoder.zero_grad()
            classifier.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")

def train_contrastive(config,encoder):
    pass

def contrastive_loss(hidden,labels,temperature=0.4,
                         weights=1.0):
    LARGE_NUM = 1e9
    hidden=torch.linalg.norm(hidden,dim=-1)
    hidden1,hidden2=torch.split(hidden, 2, dim=0)
    batch_size = hidden1.shape[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    CEL=nn.CrossEntropyLoss(weight=weights)
    # labels = torch.one_hot(torch.range(batch_size), batch_size * 2)#因为是和自己做所以只有对角线为正
    # masks = torch.one_hot(torch.range(batch_size), batch_size)#注意这个函数pytorch无

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,-1,-2)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM#这个是为什么？
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large,-1,-2)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,-1,-2)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,-1,-2)) / temperature


    loss_a = CEL(
        torch.concat([logits_ab, logits_aa], 1),labels,  weights=weights)
    loss_b = CEL(
         torch.concat([logits_ba, logits_bb], 1),labels,  weights=weights)
    loss = loss_a + loss_b

    return loss, logits_ab, labels

def train_mem_model(config, encoder, classifier, memory_network, training_data, mem_data, epochs):
    data_loader = get_data_loader(config, training_data)
    encoder.train()
    classifier.train()
    memory_network.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001},
        {'params': memory_network.parameters(), 'lr': 0.0001}
    ])

    # mem_data.unsqueeze(0)
    # mem_data = mem_data.expand(data_loader.batch_size, -1, -1)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens) in enumerate(data_loader):

            mem_for_batch = mem_data.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(len(tokens), -1, -1)

            encoder.zero_grad()
            classifier.zero_grad()
            memory_network.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            reps = memory_network(reps, mem_for_batch)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(memory_network.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")

def evaluate_model(config, encoder, classifier, memory_network, test_data, protos4eval):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    memory_network.eval()
    n = len(test_data)

    correct = 0
    protos4eval.unsqueeze(0)
    protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
    for step, (labels, tokens) in enumerate(data_loader):
        mem_for_batch = protos4eval.clone()
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        reps = memory_network(reps, mem_for_batch)
        logits = classifier(reps)

        neg_index = random.sample(range(0, 80), 10)
        neg_sim = logits[:,neg_index].cpu().data.numpy()
        max_smi = np.max(neg_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n

def evaluate_no_mem_model(config, encoder, classifier, test_data):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        neg_index = random.sample(range(0, 80), 10)
        neg_sim = logits[:,neg_index].cpu().data.numpy()
        max_smi = np.max(neg_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n

def evaluate_strict_model(config, encoder, classifier, memory_network, test_data, protos4eval,seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    memory_network.eval()
    n = len(test_data)

    correct = 0
    protos4eval.unsqueeze(0)
    protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
    for step, (labels, tokens) in enumerate(data_loader):
        mem_for_batch = protos4eval.clone()
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        reps = memory_network(reps, mem_for_batch)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n

def evaluate_strict_no_mem_model(config, encoder, classifier, test_data, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n



class sample_dataloader(object):

    def __init__(self,quadruple, id2sent,batch_size,config=None, seed=None):
        self.quadruple=quadruple
        self.id2sent=id2sent
        self.data_idx=self.get_contrastive_data(self.quadruple,batch_size)
        self.config = config
        self._ix=0
        self.to_Tensor

    def __iter__(self):
        return self

    def __next__(self):
        try:
            l_idx,is_p_data = self.data_idx[self._ix]
        except IndexError:
            # Possibly reset `self._ix`?
            raise StopIteration

            batch_all=[self.quadruple[i] for i in l_idx]
            sent_inp,emb_inp,preds,trues=[],[],[],[]
            for (sent_id,emb,pred,true) in batch_all:
                sent_inp.append(self.id2sent[sent_id])
                emb_inp.append(emb)
                preds.append(pred)
                trues.append(true)
            labels=np.zeros(shape=(len(trues),len(trues)))
            if is_p_data:
                comparison=None
                for i in range(1,len(trues)):
                    for j in range(0,i):
                        labels[i][j]=(trues[i]==trues[j])
            else:
                comparison = torch.ones((len(trues), len(trues)))
                for i in range(1, len(trues)):
                    for j in range(0, i):
                        labels[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) * (trues[i] == trues[j])
                        # AB+AB^C+A^BC
                        comparison[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) or (
                                    (preds[i] == trues[i]) * (preds[j] != trues[j]) + (preds[i] != trues[i]) * (
                                        preds[j] == trues[j])) * (trues[i] == trues[j])
            labels+=labels.T
            for i in range(len(trues)):
                labels[i][i]=(preds[i]==trues[i])
            labels=torch.IntTensor(labels)


        self._ix += 1
        return sent_inp,emb_inp,labels,comparison

    def get_contrastive_data(self,quadruple,  batch_size):
        """
        From
        1. Positive and negative embeddings in one sentence.
        2. Between cluster
        3. In cluster different sentence positive embedding.
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
        for i, quad in enumerate(quadruple):
            sent_id2ins_id[quad[0]].append(i)
            if quad[-1] == quad[-2]:
                p_idx.append(i)

        p_batch_idx.append((self.multi_sample_no_replace(p_idx, batch_size),1))
        for ins in sent_id2ins_id.values():
            n = len(ins) / batch_size
            if n > 0:
                pn_batch_idx.append((self.multi_sample_no_replace(ins, batch_size),0))
                # for i in range(round(n - 0.1)):
                #     pn_batch_idx.append(np.random.choice(a=ins, size=batch_size, replace=False).tolist())
            else:
                if n > 0.8:  # just sample when enough positive and negative samples
                    count += 1
                    print(f"Over sampling num:{count}")
                    pn_batch_idx.append((np.random.choice(a=ins, size=batch_size, replace=True).tolist(),0))
        random.shuffle(p_batch_idx.extend(pn_batch_idx))
        return p_batch_idx

    def multi_sample_no_replace(self,list_collection, n,shuffle=True):
        if shuffle:
            random.shuffle(list_collection)
        return list(self.split_list_by_n(list_collection,n,last_to_n=True))

    def split_list_by_n(self,list_collection, n,last_to_n=False):
        """
        将list均分，每份n个元素
        :return:返回的结果为评分后的每份可迭代对象
        """

        for i in range(0, len(list_collection), n):
            if last_to_n:
                if (i+n)>len(list_collection):
                    yield list_collection[i:]+random.choice(list_collection,i+n-len(list_collection)-1)
                else:
                    yield list_collection[i: i + n]

            else:
                yield list_collection[i: i + n]

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)



    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for i in range(self.config.num_of_relation)]
        val_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]
        for relation in data.keys():
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]
                tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.config.max_length)
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


    


if __name__ == '__main__':
    with open("id2sentence.pkl","rb") as f,open("quads.pkl","rb") as f1:
        id2sent,quads=pickle.load(f),pickle.load(f1)
        get_contrastive_data(quads, id2sent)
    # parser = ArgumentParser(
    #     description="Config for lifelong relation extraction (classification)")
    # parser.add_argument('--config', default='config.ini')
    # args = parser.parse_args()
    # config = Config(args.config)
    #
    # config.device = torch.device(config.device)
    # config.n_gpu = torch.cuda.device_count()
    # config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)
    #
    # # output result
    # printer = outputer()
    # middle_printer = outputer()
    # start_printer=outputer()
    #
    # # set training batch
    # for i in range(config.total_round):
    #
    #     test_cur = []
    #     test_total = []
    #
    #     # set random seed
    #     random.seed(config.seed+i*100)
    #
    #     # sampler setup
    #     sampler = data_sampler(config=config, seed=config.seed+i*100)
    #     id2rel = sampler.id2rel
    #     rel2id = sampler.rel2id
    #     # encoder setup
    #     encoder = Bert_Encoder(config=config).to(config.device)
    #     # classifier setup
    #     classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation).to(config.device)
    #
    #     # record testing results
    #     sequence_results = []
    #     result_whole_test = []
    #
    #     # initialize memory and prototypes
    #     num_class = len(sampler.id2rel)
    #     memorized_samples = {}
    #
    #     # load data and start computation
    #     for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
    #
    #         # print(current_relations)
    #         # section 1: 前向计算，算mdropout并让loss最小，然后用不确定度选择出正负样本并存入文件。因为后面涉及到对关系的对比学习，对比学习格式是：
    #         # 每个数据应该有以下几个特征：（预测类别，真实样本关系类别，样本，向量，对错）
    #         # 对比样本和向量的相似度，输入左边是样本，右边是向量。
    #         # 所以根据样本来进行采样。
    #         # 样本，向量，真实样本关系类别==预测类别
    #
    #         temp_mem = {}
    #         temp_protos = []
    #         for relation in seen_relations:
    #             if relation not in current_relations:
    #                 temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))
    #
    #         # Initial
    #         train_data_for_initial = []
    #         for relation in current_relations:
    #             train_data_for_initial += training_data[relation]
    #         # train model
    #         train_simple_model(config, encoder, classifier, train_data_for_initial, config.step1_epochs)
    #
    #
    #         # # Memory Activation
    #         # train_data_for_replay = []
    #         # random.seed(config.seed+i*100)
    #         # for relation in current_relations:
    #         #     train_data_for_replay += training_data[relation]
    #         # for relation in memorized_samples:
    #         #     train_data_for_replay += memorized_samples[relation]
    #         # train_simple_model(config, encoder, classifier, train_data_for_replay, config.step2_epochs)
    #
    #         for relation in current_relations:
    #             temp_mem[relation] = select_data(config, encoder, training_data[relation])
    #             temp_protos.append(get_proto(config, encoder, temp_mem[relation]))
    #         temp_protos = torch.cat(temp_protos, dim=0).detach()
    #
    #         memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
    #                                           input_size=encoder.output_size,
    #                                           output_size=encoder.output_size,
    #                                           key_size=config.key_size,
    #                                           head_size=config.head_size
    #                                           ).to(config.device)
    #
    #         # generate training data for the corresponding memory model (ungrouped)
    #         train_data_for_memory = []
    #         for relation in temp_mem.keys():
    #             train_data_for_memory += temp_mem[relation]
    #         for relation in memorized_samples.keys():
    #             train_data_for_memory += memorized_samples[relation]
    #         random.shuffle(train_data_for_memory)
    #         train_mem_model(config, encoder, classifier, memory_network, train_data_for_memory, temp_protos, config.step3_epochs)
    #
    #         # regenerate memory
    #         for relation in current_relations:
    #             memorized_samples[relation] = select_data(config, encoder, training_data[relation])
    #         protos4eval = []
    #         for relation in memorized_samples:
    #             protos4eval.append(get_proto(config, encoder, memorized_samples[relation]))
    #         protos4eval = torch.cat(protos4eval, dim=0).detach()
    #
    #         test_data_1 = []
    #         for relation in current_relations:
    #             test_data_1 += test_data[relation]
    #
    #         test_data_2 = []
    #         for relation in seen_relations:
    #             test_data_2 += historic_test_data[relation]
    #
    #         cur_acc = evaluate_strict_model(config, encoder, classifier, memory_network, test_data_1, protos4eval,seen_relations)
    #         total_acc = evaluate_strict_model(config, encoder, classifier, memory_network, test_data_2, protos4eval,seen_relations)
    #         # cur_acc = evaluate_strict_no_mem_model(config, encoder, classifier, test_data_1, seen_relations)
    #         # total_acc = evaluate_strict_no_mem_model(config, encoder, classifier, test_data_2, seen_relations)
    #
    #         # encoder.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_encoder_task' + str(steps+1)+'.json')
    #         # classifier.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_classifier_task' + str(steps+1)+'.json')
    #         # memory_network.save_parameters('./model_parameters/abalation_study/FewRel_10tasks_memory_network_task' + str(steps+1)+'.json')
    #         # np.save('./model_parameters/abalation_study/FewRel_10tasks_mem_task' + str(steps+1)+'.npy', protos4eval.cpu().numpy())
    #
    #         print(f'Restart Num {i+1}')
    #         print(f'task--{steps + 1}:')
    #         print(f'current test acc:{cur_acc}')
    #         print(f'history test acc:{total_acc}')
    #         test_cur.append(cur_acc)
    #         test_total.append(total_acc)
    #         print(test_cur)
    #         print(test_total)


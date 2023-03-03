# import os
# import sys
# # 找到当前文件的决定路径,__file__ 表示当前文件,也就是test.py
# file_path = os.path.abspath(__file__)
# print(file_path)
# # 获取当前文件所在的目录
# cur_path = os.path.dirname(file_path)
# print(cur_path)
# # 获取项目所在路径
# project_path = os.path.dirname(cur_path)
# print(project_path)
# # 把项目路径加入python搜索路径
# sys.path.append(project_path)
import itertools
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import copy
from scipy import linalg
import collections
import functools
import heapq
import time

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
import pickle

from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
import torch.nn.functional as F

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.dropout_layer import Dropout_Layer
from model.classifier.softmax_classifier import Softmax_Layer
from model.contrastive_network.contrastive_network import ContrastiveNetwork

from utils import outputer, batch2device, statistic_old_new

from Sampler.sample_dataloader import sample_dataloader, data_sampler, MyDataset, memory_fn
from data_loader import get_data_loader
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified
from sklearn import manifold
import matplotlib.pyplot as plt


def contrastive_loss(hidden, labels, FUNCODE=0):
    LARGE_NUM = 1e9
    # hidden=torch.linalg.norm(hidden,dim=-1)
    # hidden1,hidden2=torch.split(hidden, 2, dim=0)
    # batch_size = hidden1.shape[0]
    # hidden1_large = hidden1
    # hidden2_large = hidden2
    if FUNCODE == 0:
        logsoftmax = nn.LogSoftmax(dim=-1)
        return -(logsoftmax(hidden) * labels / labels.shape[0]).sum()
        # a=-(logsoftmax(hidden) * labels).sum() / labels.shape[0]
        # alpha1=1
        # alpha2=0.001
        # m=1
        # b=(m-torch.max(-hidden + torch.gather(hidden, 1, torch.argmax(labels, dim=-1).view(-1, 1)) , dim=1)[0]).mean()
        #
        # return alpha1*a+alpha2*b
        # ce_loss=(logsoftmax(hidden) * labels)
        # pt=torch.exp(-ce_loss)
        # return (alpha * (1 - pt) ** gamma * ce_loss).sum()/ labels.shape[0]# focal loss
    elif FUNCODE == 1:
        # alpha = 0.25
        alpha = 1
        gamma = 2
        ce_loss = torch.nn.functional.cross_entropy(hidden, torch.argmax(labels, dim=-1), reduction='none')
        # ce_loss = torch.nn.functional.cross_entropy(hidden, labels.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()
    # loss=nn.CrossEntropyLoss()
    # return loss(hidden,labels)

    # logsigmoid=nn.LogSigmoid()
    # return (-logsigmoid(hidden)*labels-(1-logsigmoid(hidden))*(1-labels)).sum()/(labels.shape[0]*labels.shape[1])
    elif FUNCODE == 2:
        logsoftmax = nn.LogSoftmax(dim=-1)
        # a=-(logsoftmax(hidden) * labels).sum() / labels.shape[0]
        alpha = 1
        gamma = 2
        ce_loss = torch.nn.functional.cross_entropy(hidden, torch.argmax(labels, dim=-1), reduction='none')
        # ce_loss = torch.nn.functional.cross_entropy(hidden, labels.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        a = (alpha * (1 - pt) ** gamma * ce_loss).mean()
        alpha1 = 1
        alpha2 = 1e-3
        m = 2
        # b=(m-torch.max(-hidden + torch.gather(hidden, 1, torch.argmax(labels, dim=-1).view(-1, 1)) , dim=1)[0]).mean
        tmp, _ = torch.topk(hidden, 2, dim=1, largest=True, sorted=True)
        tt = -tmp[:, 0] + tmp[:, 1] + m
        b = (tt[tt >= 0]).sum() / tmp.shape[0]
        return alpha1 * a + alpha2 * b


def distill_loss(pre, cur):
    return -torch.mean(torch.sum(F.softmax(pre, dim=1) * F.log_softmax(cur, dim=1), dim=1))


# def compute_jsd_loss(m_input):
#     # m_input: the result of m times dropout after the classifier.
#     # size: m*B*C
#     m = m_input.shape[0]
#     mean = torch.mean(m_input, dim=0)# BC
#     jsd = 0
#     for i in range(m):
#         loss = F.kl_div(F.softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
#         loss = loss.sum()
#         jsd += loss / m
#     return jsd
def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def train_contrastive(config, logger, model, optimizer, scheduler, dataloader, evaluator, encoder, dropout_layer,
                      memory_network=None, mem_data=None, epoch=None, n_rel=None, distilling=False):
    train_dataloader = dataloader
    epoch = epoch or config.contrast_epoch
    # if distilling:
    #     pre_enc=copy.deepcopy(encoder)
    #     pre_drop=copy.deepcopy(dropout_layer)
    #     pre_model=copy.deepcopy(model)
    last_acc = -1
    count = 0
    ep = 0

    while True:
        ## train
        model.train()
        t_ep = time.time()
        # epoch parameter start
        batch_cum_loss, batch_cum_right, batch_cum_len = 0., 0., 0.

        # epoch parameter end
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            # batch parameter start
            t_batch = time.time()
            # batch parameter end
            # move to device
            # if not config.fix_sent:
            # batch_train_data = batch2device(batch_train_data, config.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            sent_inp, emb_inp, labels, comparison, sentence_labels = batch_train_data

            labels = labels.to(config.device)
            emb_inp = emb_inp.to(config.device)

            is_old, is_new = [], []
            for i, sent_label in enumerate(sentence_labels.tolist()):
                if sent_label not in n_rel:
                    is_old.append(i)
                else:
                    is_new.append(i)
            if config.CDE == 0:  # only old
                if len(is_old) != 0:
                    old_sent = sent_inp[is_old, :]
                    old_labels = labels[is_old, :]
                    old_labels = old_labels.repeat(config.f_pass, 1)
                    old_sent = old_sent.to(config.device)
                    # old sent
                    enc_inp = encoder(old_sent)  # m Old P
                    results_old = []
                    for _ in range(config.f_pass):
                        hidden = model(dropout_layer(enc_inp)[1], emb_inp, FUN_CODE=4)
                        results_old.append(hidden)
                    results_old = torch.cat(results_old, dim=0)
                    # loss3=compute_jsd_loss(results_old)
                else:
                    results_old = []
                    old_labels = []
                    # loss3=0

                # new sent
                if len(is_new) != 0:
                    new_sent = sent_inp[is_new, :]
                    new_labels = labels[is_new, :]
                    new_sent = new_sent.to(config.device)
                    # new sent
                    left = dropout_layer(encoder(new_sent))[1]
                    result_new = model(left, emb_inp, FUN_CODE=4)  # New P
                    # cat

                    # # distilling
                    # if distilling:
                    #     with torch.no_grad():
                    #         prev = pre_model(pre_drop(pre_enc(new_sent))[1], emb_inp, FUN_CODE=4)  # previous
                    #     loss2 = distill_loss(prev, result_new)
                    # else:
                    #     loss2=0
                else:
                    result_new = []
                    new_labels = []
                    # loss2=0
                if len(old_labels) and len(new_labels):
                    labels = torch.cat([old_labels, new_labels], dim=0)
                    hidden = torch.cat([results_old, result_new], dim=0)
                elif len(old_labels):
                    labels = old_labels
                    hidden = results_old
                elif len(new_labels):
                    labels = new_labels
                    hidden = result_new

                # results=torch.stack(results,dim=0)
                # model forward end
                # grad operation start

                loss1 = contrastive_loss(hidden, labels, FUNCODE=2)
                # print(f"loss1/0.01*loss2:{loss1/(0.001*loss2)}")
                # a,b=len(is_old),len(is_new)
                # alpha1=(config.f_pass*a+2*b)/(config.f_pass*a+b)
                # alpha2=(config.f_pass*a+2*b)/(b)
                loss = loss1
                loss.backward()
                optimizer.step()
                # if not config.fix_sent:
                #     loss = contrastive_loss(hidden, labels, FUNCODE=0)
                # else:
                #     loss = contrastive_loss(hidden, labels, FUNCODE=2)
                # loss.backward()
                # optimizer.step()
            elif config.CDE == 1:  # enhance all
                sent_inp = sent_inp.to(config.device)
                enc_inp = encoder(sent_inp)  # m Old P
                results = []
                for _ in range(config.f_pass):
                    hidden = model(dropout_layer(enc_inp)[1], emb_inp, FUN_CODE=4)
                    results.append(hidden)
                hidden = torch.cat(results, dim=0)
                # loss3 = compute_jsd_loss(hidden)
                labels = labels.repeat(config.f_pass, 1)
                loss = contrastive_loss(hidden, labels, FUNCODE=2)
                loss.backward()
                optimizer.step()
            elif config.CDE == 2:  # no enhance
                optimizer.zero_grad()
                batch_train_data = batch2device(batch_train_data, config.device)
                # zero the parameter gradients
                sent_inp, emb_inp, labels, comparison, sentence_labels = batch_train_data
                enc_inp = encoder(sent_inp)  # m Old P
                hidden = model(dropout_layer(enc_inp)[1], emb_inp, FUN_CODE=4)
                loss = contrastive_loss(hidden, labels, FUNCODE=2)
                loss.backward()
                optimizer.step()
            # grad operation end

            # accuracy calculation start
            # acc = ((softmax(hidden) > 0.5) == labels).sum() / comparison.sum()
            right = (torch.argmax(hidden, dim=-1) == torch.argmax(labels, dim=-1)).sum()
            acc = right / hidden.shape[0]

            loss, acc = loss.item(), acc.item()

            batch_cum_len += hidden.shape[0]
            batch_cum_loss += loss
            batch_cum_right += right.item()

            batch_avg_loss = batch_cum_loss / (batch_ind + 1)
            batch_avg_acc = batch_cum_right / batch_cum_len
            # accuracy calculation end
            batch_print_format = "\rContrastive Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "acc: {}, " + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                ep + 1,
                epoch,
                batch_ind + 1,
                len(train_dataloader),
                batch_avg_loss,
                batch_avg_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")
            # batch logger and print end

            # change lr
            scheduler.step()
        ep += 1
        cont_total_acc, topk_total_acc = evaluate_contrastive_model(config, dropout_layer, encoder, contrastive_network,
                                                                    memory,
                                                                    test_data_2,
                                                                    use_mem_net=False,
                                                                    new_rel=n_rel)
        print(f"test acc:{cont_total_acc}")

        if int(1e5 * last_acc) < int(1e5 * cont_total_acc):
            count = 0
        else:
            count += 1
        last_acc = cont_total_acc
        if count >= 4 and batch_avg_acc >= 0.9:
            break
        # if ep==10:
        #     break
        # epoch logger and print start
        # logger.log({
        #     "train_loss": batch_avg_loss,
        #     "train_ent_seq_acc": batch_avg_acc,
        #     "learning_rate": optimizer.param_groups[0]['lr'],
        #     "time": time.time() - t_ep,
        # })
        # epoch logger and print start






# def computer_CrossEntropyLoss(predicts,labels):
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def select_to_memory(config, encoder, dropout_layer, classifier, training_data, memory):
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()

    with torch.no_grad():
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            # with torch.no_grad():
            tokens_id = torch.stack(tokens_id, dim=0)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)  # b,s
            labels = labels.to(config.device)  # b
            reps = encoder(tokens)
            output, output_embedding_true = dropout_layer(reps)  # B H
            logits = classifier(output)
            # model prediction
            out_prob = F.softmax(logits, dim=-1)
            max_idx_true = torch.argmax(out_prob, dim=-1)  # B

            enable_dropout(dropout_layer)
            out_prob = []
            logits_all = []
            for _ in range(config.f_pass):
                output, _ = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))

            out_prob = torch.stack(out_prob)  # m b c
            out_std = torch.std(out_prob, dim=0)  # B,C

            labels_mask = torch.zeros_like(out_std).scatter_(-1, labels.view(
                labels.shape[0], 1), 1)  # B,C
            preds_mask = torch.zeros_like(out_std).scatter_(-1, max_idx_true.view(
                max_idx_true.shape[0], 1), 1)  # B,C
            # uncertainty
            slt_mask = (out_std < config.kappa_pos) * (labels_mask * preds_mask)  # 不满足被标志为1之后去除 #BC

            slt_idx = torch.sum(slt_mask, dim=-1) > 0  # B,C->B
            slt_tokens_ids, slt_embeddings, slt_preds, slt_labels = tokens_id[slt_idx], output_embedding_true[slt_idx], \
                                                                    max_idx_true[slt_idx], labels[slt_idx]
            slt_unct = torch.sum(out_std * slt_mask, dim=-1)[slt_idx]  #
            # data store
            # quadruple.extend(tokens)
            # index range:[step * data_loader.batch_size + id,step * data_loader.batch_size + len(tokens))

            config.K = 10  # TODO!!!!!!!!!!!!!!!!!!!!!
            for i in range(len(slt_tokens_ids)):
                memory_list = memory[slt_labels[i].item()]
                k = len(memory_list)
                heapq.heappush(memory_list,
                               (-slt_unct[i].item(), (
                                   slt_tokens_ids[i].tolist(), slt_embeddings[i].cpu(), slt_preds[i].item(),
                                   slt_labels[i].item())))
                while k > config.K:
                    heapq.heappop(memory_list)
                    k -= 1
    return memory


def slt_to_mem_by_uncty(config, encoder, dropout_layer, classifier, training_data, current_relations, rel2id, fix_label,
                        n_rel=None):
    r_id2id = relation2id(convert_relations=current_relations, rel2id=rel2id)
    id2rel_id = {v: k for k, v in r_id2id.items()}
    if not fix_label:
        n_rel = set(r_id2id[i] for i in n_rel)
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    epochs = config.train_epoch
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    cur_num = np.array([0, 0, 0, 0, 0, 0])
    ret_d = []
    with torch.no_grad():
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            out_prob = []
            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            if not fix_label:
                labels = labels.tolist()
                labels = torch.tensor([r_id2id[label] for label in labels], device=config.device)
            labels = labels.to(config.device)  # B labels don't need to care

            reps = encoder(tokens)
            enable_dropout(dropout_layer)
            for _ in range(config.f_pass):
                output, _ = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))
            out_prob = torch.stack(out_prob)  # mBC
            logits_all = torch.stack(logits_all)  # mBC
            out_std = torch.std(out_prob, dim=0)  # BC

            uncty = torch.gather(out_std, 1, labels.view(-1, 1)).reshape(-1)  # B
            # out_std_all = out_std.expand((config.f_pass, out_std.shape[0], -1))  # mBC
            # uncertainty
            # 众数投票 统计正确数，正确数，优先选正确数少的，如果正确数一致了，选不确定度高的。
            # 每个样本，统计得到（不正确数，不确定度）
            # out_std_mask_one = out_std < config.kappa_pos  # 不满足被标志为1之后去除 #BC #不确定度很低的有哪些
            # out_std_mask = out_std_mask_one.expand((config.f_pass, out_std_mask_one.shape[0], -1))  # mBC
            max_idx = torch.argmax(out_prob, dim=-1)  # m,B
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            slt_masks = (max_idx == m_labels).sum(dim=0)  # B

            num = statistic_old_new(max_idx, m_labels, new_rel=n_rel)
            cur_num += num
            # score = torch.stack([slt_masks, uncty], dim=1).tolist()  # B*2
            slt_masks = slt_masks.tolist()  # B
            uncty = uncty.tolist()  # B
            if not fix_label:
                labels = [id2rel_id[label] for label in labels.tolist()]
            else:
                labels = labels.tolist()
            tokens = tokens.tolist()

            for i in range(len(slt_masks)):
                memory_list = memory[labels[i]]
                heapq.heappush(memory_list, (-slt_masks[i], uncty[i], i,
                                             {"tokens_id": tokens_id[i].item(), "relation": labels[i],
                                              "tokens": tokens[i]}))
                k = len(memory_list)
                while k > config.num_protos:
                    heapq.heappop(memory_list)
                    k -= 1
        print(
            f"valid:(too,tnn,foo,fnn,fon,fno):{cur_num},old,new pred error rate:{((cur_num[2] + cur_num[4]) / (cur_num[0] + cur_num[2] + cur_num[4]), (cur_num[3] + cur_num[5]) / (cur_num[3] + cur_num[1] + cur_num[5]))}")

    return memory  # just for remain of memory has change


def train_first(config, encoder, dropout_layer, classifier, training_data, cum_relations, rel2id, fix_label,
                FUNCODE=0, n_rel=None):
    r_id2id = relation2id(convert_relations=cum_relations, rel2id=rel2id)
    if not fix_label:
        n_rel = set(r_id2id[i] for i in n_rel)
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    epochs = config.train_epoch
    encoder.train()
    dropout_layer.train()
    classifier.train()
    cur_num = np.array([0, 0, 0, 0, 0, 0])
    criterion = nn.CrossEntropyLoss()
    # criterion = functools.partial(contrastive_loss, FUNCODE=1)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 5e-5},
        {'params': dropout_layer.parameters(), 'lr': 5e-4},
        {'params': classifier.parameters(), 'lr': 5e-3}
    ])
    id2sent = []
    ret_d = []
    for epoch_i in range(epochs):
        losses = []
        batch_cum_loss, batch_cum_acc = 0., 0.
        t_ep = time.time()
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            # with torch.no_grad():
            t_batch = time.time()
            optimizer.zero_grad()
            out_prob = []
            logits_all = []

            tokens_id = torch.stack(tokens_id, dim=0)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            if not fix_label:
                labels = labels.tolist()
                labels = torch.tensor([r_id2id[label] for label in labels], device=config.device)
            labels = labels.to(config.device)

            reps = encoder(tokens)
            output_embeddings = []
            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                if epoch_i == epochs - 1:
                    output_embeddings.append(output_embedding)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))
            out_prob = torch.stack(out_prob)  # m,B,C
            logits_all = torch.stack(logits_all)
            out_std = torch.std(out_prob, dim=0)  # BC
            # out_std_all = []
            # for i in range(config.f_pass):
            #     out_std_all.append(out_std)#xx
            # out_std_all = torch.stack(out_std_all)
            out_std_all = out_std.expand((config.f_pass, out_std.shape[0], -1))  # mBC
            # uncertainty
            out_std_mask_one = out_std < config.kappa_pos  # 不满足被标志为1之后去除 #BC #不确定度很低的有哪些
            out_std_mask = out_std_mask_one.expand((config.f_pass, out_std_mask_one.shape[0], -1))  # m B C
            max_idx = torch.argmax(out_prob, dim=-1)  # m,B
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B

            # slt_labels=m_labels[(out_std_mask.sum(dim=-1)>0)] #对吗？

            # batch_p = torch.index_select(m_labels, 0, p_index)    # B
            # batch_n=(n_mask.sum(dim=-1) > 0).nonzero()
            # logits_all[out_std_mask]=-float('inf')
            # batch_p = torch.index_select(m_labels, 0, p_index)

            # labels=labels*out_std_mask
            # loss1 = criterion(slt_logits, slt_labels)
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            # loss1=contrastive_loss(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1),FUNCODE=FUNCODE)
            # logits_all.requires_grad_()
            # for i in range(config.f_pass):
            #     loss1 += criterion(logits_all[i][out_std_mask], labels)
            # loss2 = compute_jsd_loss(slt_logits)
            loss2 = compute_jsd_loss(logits_all)
            loss = loss1 + loss2
            loss.backward()
            losses.append(loss.item())
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()

            # acc=(torch.argmax(slt_logits, dim=-1) == slt_labels).sum()/(config.f_pass*len(labels))
            m_preds = torch.argmax(logits_all, dim=-1)
            acc = (m_preds == m_labels).sum() / (config.f_pass * len(labels))
            batch_cum_loss += loss
            batch_cum_acc += acc

            batch_avg_loss = batch_cum_loss / (step + 1)
            batch_avg_acc = batch_cum_acc / (step + 1)

            num = statistic_old_new(m_preds, m_labels, new_rel=n_rel)
            cur_num += num
            batch_print_format = "\rFirst Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "acc: {}, " + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                epoch_i + 1,
                config.train_epoch,
                step + 1,
                len(data_loader),
                batch_avg_loss,
                batch_avg_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")

            # if FUNCODE != 0:
            #     # data store
            #     if epoch_i == epochs - 1:
            #
            #         with torch.no_grad():
            #             labels_mask = torch.zeros_like(out_std_mask).scatter_(-1,
            #                                                                   m_labels.view(m_labels.shape[0], -1, 1),
            #                                                                   1)  # m,B,C
            #             preds_mask = torch.zeros_like(out_std_mask).scatter_(-1, max_idx.view(max_idx.shape[0], -1, 1),
            #                                                                  1)  # m,B,C
            #
            #             # slt_mask=out_std_mask.sum(dim=-1) > 0 #m B
            #             p_mask = labels_mask * preds_mask * out_std_mask  # m,B,C
            #             n_mask = ~labels_mask * preds_mask * out_std_mask  # m,B,C #没预测对且预测值类别的不确定度很高
            #             # torch.index_select(x, 0, indices)
            #             p_index = (p_mask.sum(dim=-1) > 0)  # m B
            #             n_index = (n_mask.sum(dim=-1) > 0)  # m B
            #
            #             p_labels, n_labels = m_labels[p_index], m_labels[n_index]
            #             p_logits, n_logits = logits_all[p_index], logits_all[n_index]
            #             slt_labels = torch.cat([p_labels, n_labels], dim=0)
            #             slt_logits = torch.cat([p_logits, n_logits], dim=0)
            #             # id2sent.extend(tokens.cpu())
            #             # index range:[step * data_loader.batch_size + id,step * data_loader.batch_size + len(tokens))
            #             m_tokens_ids = tokens_id.expand((config.f_pass, tokens_id.shape[0]))  # m,B
            #
            #             # m_tokens_ids = torch.Tensor(
            #             #     range(step * data_loader.batch_size, step * data_loader.batch_size + len(tokens))).expand(
            #             #     config.f_pass, -1)  # m B
            #             m_embeddings = torch.stack(output_embeddings)  # m B H
            #             slt_tokens_ids = torch.cat([m_tokens_ids[p_index], m_tokens_ids[n_index]], dim=0).int().tolist()
            #             slt_embeddings = torch.cat([m_embeddings[p_index], m_embeddings[n_index]], dim=0).cpu()
            #             slt_preds = torch.cat([max_idx[p_index], max_idx[n_index]], dim=0).tolist()
            #             slt_labels = slt_labels.tolist()
            #             for i in range(len(slt_tokens_ids)):
            #                 ret_d.append((slt_tokens_ids[i], slt_embeddings[i], slt_preds[i], slt_labels[i]))

        print(
            f"train:(too,tnn,foo,fnn,fon,fno):{cur_num},old,new pred error rate:{((cur_num[2] + cur_num[4]) / (cur_num[0] + cur_num[2] + cur_num[4]), (cur_num[3] + cur_num[5]) / (cur_num[3] + cur_num[1] + cur_num[5]))}")
        cur_num = np.array([0, 0, 0, 0, 0, 0])
        print(f"loss is {np.array(losses).mean()}")

    # with torch.no_grad():
    #     for step, (labels, tokens, tokens_id) in enumerate(data_loader):
    #         id2sentence = {}
    #         result = {}
    #         for id in range(len(tokens)):
    #             id2sentence[step * data_loader.batch_size + id] = tokens[id].detach().numpy().tolist()
    #         tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
    #         reps = encoder(tokens)
    #         for _ in range(config.f_pass):
    #             output, output_embedding = dropout_layer(reps)
    #             logits = classifier(output)
    #             out_prob = F.softmax(logits, dim=-1)
    #             max_value, max_idx = torch.max(out_prob, dim=-1)
    #             output_embedding, max_idx = output_embedding.cpu(), max_idx.cpu()
    #             # output_embedding=output_embedding.detach().numpy().tolist()
    #             # max_idx=max_idx.detach().numpy().tolist()
    #             # labels=labels.detach().numpy().tolist()
    #             for id in range(len(labels)):
    #                 mid = [step * data_loader.batch_size + id, output_embedding[id], max_idx[id], labels[id]]
    #                 if labels[id] not in result:
    #                     result[labels[id]] = [mid]
    #                 else:
    #                     result[labels[id]].append(mid)

    return ret_d


def train_last(config, encoder, dropout_layer, classifier, training_data):
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    epochs = config.train_epoch
    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-5},
        {'params': dropout_layer.parameters(), 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': 1e-6}
    ])

    for epoch_i in range(epochs):
        losses = []
        batch_cum_loss, batch_cum_acc = 0., 0.
        t_ep = time.time()
        for step, (labels, tokens, _) in enumerate(data_loader):
            t_batch = time.time()
            optimizer.zero_grad()
            logits_all = []

            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)

            reps = encoder(tokens)

            for _ in range(config.f_pass):
                output, _ = dropout_layer(reps)

                logits = classifier(output)
                logits_all.append(logits)

            logits_all = torch.stack(logits_all)

            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B

            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))

            loss2 = compute_jsd_loss(logits_all)
            loss = loss1 + loss2
            loss.backward()
            losses.append(loss.item())
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()

            acc = (torch.argmax(logits_all, dim=-1) == m_labels).sum() / (config.f_pass * len(labels))
            batch_cum_loss += loss
            batch_cum_acc += acc

            batch_avg_loss = batch_cum_loss / (step + 1)
            batch_avg_acc = batch_cum_acc / (step + 1)
            batch_print_format = "\rMemory Refine Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "acc: {}, " + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                epoch_i + 1,
                epochs,
                step + 1,
                len(data_loader),
                batch_avg_loss,
                batch_avg_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")


def transfer_to_device(list_ins, device):
    import torch
    for ele in list_ins:
        if isinstance(ele, list):
            for x in ele:
                x.to(device)
        if isinstance(ele, torch.Tensor):
            ele.to(device)
    return list_ins


# Done
def get_proto(config, encoder, dropout_layer, mem_set):
    # aggregate the prototype set for further use.
    encoder.eval()
    dropout_layer.eval()
    data_loader = get_data_loader(config, mem_set, False, False, 1)

    features = []
    with torch.no_grad():
        for step, (labels, tokens, _) in enumerate(data_loader):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature = dropout_layer(encoder(tokens))[1]
            features.append(feature)
        features = torch.cat(features, dim=0)
        proto = torch.mean(features, dim=0, keepdim=True).cpu()
        # proto=F.normalize(proto, dim=-1, p=2)

    # return the averaged prototype
    return proto


# tsne plot
def get_tsne(feature):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    x1 = tsne.fit_transform(feature)
    x1_min, x1_max = x1.min(0), x1.max(0)
    x1_norm = (x1 - x1_min) / (x1_max - x1_min)
    return x1_norm


def valid_classification(config, encoder, dropout_layer, classifier, dataset, fix_labels):
    data_loader1 = get_data_loader(config, dataset, batch_size=config.batch_size, shuffle=True)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = 0
    right = 0
    for step, (labels, tokens, _) in enumerate(data_loader1):
        with torch.no_grad():
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)

            if not fix_labels:
                labels, _ = relation2id(current_relations, rel2id, labels)

            labels = labels.to(config.device)
            reps = encoder(tokens)
            output, _ = dropout_layer(reps)
            logits = classifier(output)
            right += (torch.argmax(logits, dim=-1) == labels).sum()
            n += len(labels)

    return right / n


def tsne_plot(config, encoder, dropout_layer, tokens_task1, tokens_task2, flag, num_points):
    data_loader1 = get_data_loader(config, tokens_task1, shuffle=False, drop_last=False, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    features1 = {}
    for step, (labels, tokens, _) in enumerate(data_loader1):
        with torch.no_grad():
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            feature = dropout_layer(encoder(tokens))[1].cpu()
            labels = int(labels)
        if labels not in features1:
            features1[labels] = [feature]
        else:
            features1[labels].append(feature)
    for key, values in features1.items():
        features1[key] = get_tsne(np.concatenate(values))
    data_loader2 = get_data_loader(config, tokens_task2, shuffle=False, drop_last=False, batch_size=1)
    features2 = {}
    for step, (labels, tokens, _) in enumerate(data_loader2):
        with torch.no_grad():
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            feature = dropout_layer(encoder(tokens))[1].cpu()
            labels = int(labels)
        if labels not in features2:
            features2[labels] = [feature]
        else:
            features2[labels].append(feature)
    for key, values in features2.items():
        features2[key] = get_tsne(np.concatenate(values))
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'pink']
    if not flag:
        for i, (key, values) in enumerate(features1.items()):
            plt.scatter(values[:num_points, 0], values[:num_points, 1], c=colors[i], label='task1_' + str(key),
                        marker='.')
        for i, (key, values) in enumerate(features2.items()):
            plt.scatter(values[:num_points, 0], values[:num_points, 1], c=colors[i + len(colors) // 2],
                        label='task2_' + str(key), marker='x')
        # plt.legend()
        plt.title('distribution after first training')
        print('first picture finished')
        plt.savefig('firstpicture_2.png')
        plt.clf()
        plt.cla()
    if flag:
        for i, (key, values) in enumerate(features1.items()):
            plt.scatter(values[:num_points, 0], values[:num_points, 1], c=colors[i], label='task1_' + str(key),
                        marker='.')
        for i, (key, values) in enumerate(features2.items()):
            plt.scatter(values[:num_points, 0], values[:num_points, 1], c=colors[i + len(colors) // 2],
                        label='task2_' + str(key), marker='x')
        # plt.legend()
        plt.title('distribution after contrastive training')
        print('second picture finished')
        plt.savefig('secondpicture_2.png')

    # # Use K-Means to select what samples to save, similar to at_least = 0


def select_data(config, encoder, dropout_layer, sample_set):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, (labels, tokens, _) in enumerate(data_loader):
        with torch.no_grad():
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            feature = dropout_layer(encoder(tokens))[1].cpu()
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

    # version 3
    # features = features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)
    # km = KMeans(n_clusters=num_clusters, init='k-means++')
    # km.fit(features)
    # center = np.average(km.cluster_centers_, axis=0).reshape(1, -1)
    # d = cdist(features, center, metric='euclidean').reshape(-1)
    # center = center.reshape(-1)
    # c_labels = km.labels_
    # mem_set = []
    # for k in range(num_clusters):
    #     sel_index = np.argmax(d*(c_labels==k))
    #     instance = sample_set[sel_index]
    #     mem_set.append(instance)
    # center = center.reshape(1,-1)
    # return mem_set, F.normalize(torch.tensor(center), dim=-1, p=2)
    # version 2
    # features = features / np.linalg.norm(features, ord=2, axis=-1, keepdims=True)
    # km = KMeans(n_clusters=num_clusters, init='k-means++')
    # km.fit(features)
    # center = np.average(km.cluster_centers_, axis=0).reshape(1, -1)
    # d = cdist(features, center, metric='euclidean').reshape(-1)
    # center = center.reshape(-1)
    # c_labels = km.labels_
    # mem_set = []
    # for k in range(num_clusters):
    #     sel_index = np.argmax(d*(c_labels==k))
    #     instance = sample_set[sel_index]
    #     mem_set.append(instance)
    # center = center.reshape(1,-1)
    # return mem_set, F.normalize(torch.tensor(center), dim=-1, p=2)
    # # version 1



# def PCA(config,rel_rep):
#     from sklearn.decomposition import PCA
#     feature=torch.cat(list(rel_rep.values()),dim=0)
#     pca = PCA(n_components=feature.shape[1])
#     newX = pca.fit_transform(feature)
#     newX=torch.tensor(newX,device=config.device)
#     for i,k in enumerate(rel_rep.keys()):
#         rel_rep[k]=newX[i]
#     return rel_rep

def relation2id(convert_relations, rel2id):
    """
    :param current_relations: current relation set
    :param rel2id: all relation to id map
    :param convert_list: needed convert list
    :return: use rel2id to convert current relation to corresponding map.
    """
    d = {}  # rel_id2id
    for i in range(len(convert_relations)):
        d[rel2id[convert_relations[i]]] = i
    return d

    # convert_list = np.array(convert_list)
    # for j in range(len(convert_list)):
    #     ans.append(d[convert_list[j]])
    # return torch.LongTensor(torch.tensor(ans))


def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, cum_relations, rel2id,
                       fix_label,
                       FUNCODE=2):
    r_id2id = relation2id(convert_relations=cum_relations, rel2id=rel2id)
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 5e-5},
        {'params': dropout_layer.parameters(), 'lr': 5e-4},
        {'params': classifier.parameters(), 'lr': 5e-3}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=decay_steps,
                                                gamma=decay_rate)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            optimizer.zero_grad()
            if not fix_label:
                labels = labels.tolist()
                labels = torch.tensor([r_id2id[label] for label in labels], device=config.device)
            labels = labels.to(config.device)

            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
        print(f"loss is {np.array(losses).mean()}")


def train_mem_model(config, encoder, classifier, memory_network, training_data, mem_data, epochs):
    data_loader = get_data_loader(config, training_data)
    encoder.train()
    classifier.train()
    memory_network.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': 0.001},
        {'params': memory_network.parameters(), 'lr': 0.0001}
    ])

    # mem_data.unsqueeze(0)
    # mem_data = mem_data.expand(data_loader.batch_size, -1, -1)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            mem_for_batch = mem_data.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(len(tokens), -1, -1)

            encoder.zero_grad()
            classifier.zero_grad()
            memory_network.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
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
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        mem_for_batch = protos4eval.clone()
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps = memory_network(reps, mem_for_batch)
        logits = classifier(reps)

        neg_index = random.sample(range(0, 80), 10)
        neg_sim = logits[:, neg_index].cpu().data.numpy()
        max_smi = np.max(neg_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct / n


def evaluate_no_mem_model(config, encoder, dropout_layer, classifier, test_data):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.train()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)

    cum_acc = 0
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)
        max_idx = torch.argmax(logits, dim=-1)
        cum_acc += (max_idx == labels).sum() / labels.shape[0]

    return cum_acc / len(data_loader)


def evaluate_strict_model(config, encoder, classifier, memory_network, test_data, protos4eval, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    memory_network.eval()
    n = len(test_data)

    correct = 0
    protos4eval.unsqueeze(0)
    protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        mem_for_batch = protos4eval.clone()
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps = memory_network(reps, mem_for_batch)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1
    return correct / n


def evaluate_strict_no_mem_model(config, encoder, classifier, test_data, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct / n


def evaluate_first_model(config, encoder, dropout_layer, classifier, test_data, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)
    correct = 0
    # cum_acc = 0
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)
        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1
    return correct / n


def evaluate_contrastive_model(config, dropout_layer, encoder, contrastive_network, memory, test_data,
                               use_mem_net=False, memory_network=None,
                               protos4eval=None, test_emb=None, new_rel=None):
    batch_size = 2
    testdata_loader = get_data_loader(config, test_data, batch_size=batch_size)
    if use_mem_net:
        memory_network.eval()
    contrastive_network.eval()
    encoder.eval()
    dropout_layer.eval()
    # protos4eval= protos4eval.unsqueeze(0)
    # protos4eval = protos4eval.expand(batch_size + config.num_protos, -1, -1)  # slow!!!!!!!!!!!!!!
    cur_num = np.array([0, 0, 0, 0, 0, 0])
    cum_right = 0
    cum_len = 0
    # top_k_right = 0
    if test_emb is None:
        test_emb = {}
        for label_id, ins_list in memory.items():
            test_emb[label_id] = get_proto(config, encoder, dropout_layer, ins_list)
    mem_id2label = list(test_emb.keys())
    label2mem_id = {lb: i for i, lb in enumerate(mem_id2label)}
    if new_rel:
        new_rel = set(label2mem_id[i] for i in new_rel)
    with torch.no_grad():
        right = torch.cat(list(test_emb.values()), dim=0).to(config.device)  # B2*H
        for step, (labels, tokens, _) in enumerate(testdata_loader):
            # results = torch.zeros(len(tokens), len(memory), device=config.device)  # B1 B2
            labels = torch.stack([torch.tensor(label2mem_id[label.item()], device=config.device) for label in labels],
                                 dim=-1)
            # labels = labels.to(config.device)  # B1
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)  # B1*H
            # enable_dropout(dropout_layer)
            # enc_inp = encoder(tokens)  # m Old P
            # out_prob = []
            # for _ in range(config.f_pass):
            #     hidden = dropout_layer(enc_inp)[1]
            #     logits_aa = contrastive_network(hidden, right, FUN_CODE=4)
            #     out_prob.append(F.softmax(logits_aa, dim=-1))
            # out_prob = torch.stack(out_prob)  # m,B,C
            # out_std = torch.std(out_prob, dim=0)  # BC
            # dropout_layer.eval()
            tokens = dropout_layer(encoder(tokens))[1]
            if use_mem_net:
                mem_for_batch = protos4eval.clone()
                mem_for_batch.unsqueeze(0)
                mem_for_batch = mem_for_batch.expand(len(tokens) + len(right), -1, -1)
                logits_aa = contrastive_network(tokens, right, comparison=torch.ones(len(tokens), len(memory),
                                                                                     device=config.device),
                                                memory_network=memory_network, mem_for_batch=mem_for_batch,
                                                FUN_CODE=3)  # B1*B2
            else:
                logits_aa = contrastive_network(tokens, right, FUN_CODE=4)  # B1*B2
            # predict_matrix = torch.argmax(logits_aa, dim=-1)  # B1
            results = logits_aa
            # _, predict_matrix = torch.topk(logits_aa, k=2, dim=-1, largest=True)  # B1
            # results += torch.zeros_like(results).scatter_(-1, predict_matrix, 1)
            # for j, i in enumerate(predict_matrix.tolist()):
            #     results[j][i] += 1

            # preds = torch.stack(
            #     [torch.tensor(mem_id2label[i], device=config.device) for i in torch.argmax(results, dim=-1)], dim=-1)  # B1
            # torch.gather(results)
            preds = torch.argmax(results, dim=-1)  # B
            # _, topk_preds = torch.topk(results, k=2, dim=-1, largest=True)  # B K
            # topk_mask = torch.zeros((preds.shape[0], len(label2mem_id)), device=config.device).scatter_(-1, topk_preds,
            #                                                                                             1)
            labels_mask = torch.zeros((preds.shape[0], len(label2mem_id)), device=config.device).scatter_(-1,
                                                                                                          labels.view(
                                                                                                              -1, 1), 1)
            # top_k_right += (topk_mask * labels_mask).sum().item()
            cum_right += (labels == preds).sum().item()
            cum_len += len(preds)
            num = statistic_old_new(preds, labels, new_rel=new_rel)
            cur_num += num
    print(f"Contrastive acc is {cum_right / cum_len}")
    print(
        f"test:(too,tnn,foo,fnn,fon,fno):{cur_num},old,new pred error rate:{((cur_num[2] + cur_num[4]) / (cur_num[0] + cur_num[2] + cur_num[4]), (cur_num[3] + cur_num[5]) / (cur_num[3] + cur_num[1] + cur_num[5]))}")
    return cum_right / cum_len, None


def quads2origin_data(quads, id2sentence):
    ret_d = []
    for quad in quads:
        tokenized_sample = {}
        tokenized_sample["tokens_id"] = quad[0]
        tokenized_sample['relation'] = quad[-1]
        tokenized_sample['tokens'] = id2sentence[quad[0]]
        ret_d.append(tokenized_sample)
    return ret_d


def trans_memory_format(memory, cur_rel_id):
    for k in cur_rel_id:
        new = []
        for i in memory[k]:
            new.append(i[3])
        memory[k] = new
    return memory


if __name__ == '__main__':
    FUNCODE = 3  # Funcode==1为4类版本，为2为多类版本

    use_mem_network = False
    fix_labels = False
    parser = ArgumentParser(
        description="Config for lifelong relation extraction (classification)")
    d_pic = False
    verify_history = False
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)
    config.fix_classes = fix_labels
    config.use_atm = use_mem_network
    config.fix_sent = True
    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    # output result
    # printer = outputer()
    # middle_printer = outputer()
    # start_printer = outputer()

    # set training batch
    all_results = []
    all_results_cur = []
    for i in range(config.total_round):
        print(f"total round {i}")
        test_cur = []
        test_total = []
        # set random seed
        random.seed(config.seed + i * 100)

        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100)
        id2sentence = sampler.get_id2sent()
        # with open('id2sentence.pkl', 'wb') as f:
        #     pickle.dump(id2sentence, f)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        # encoder setup
        encoder = Bert_Encoder(config=config).to(config.device)
        # encoder1 = Bert_Encoder(config=config).to(config.device)
        # dropout setup
        dropout_layer = Dropout_Layer(config=config, input_size=encoder.output_size).to(config.device)
        # classifier setup
        if not fix_labels:
            num_class = config.rel_per_task
        else:
            num_class = config.num_of_relation

        classifier = Softmax_Layer(input_size=encoder.output_size, num_class=num_class).to(config.device)
        # 这里的encoder没有加dropout_layer
        contrastive_network = ContrastiveNetwork(config=config,
                                                 hidden_size=config.encoder_output_size).to(
            config.device)
        # record testing results
        sequence_results = []
        result_whole_test = []
        # T_mult = config.T_mult
        # rewarm_epoch_num = config.rewarm_epoch_num
        decay_rate = config.decay_rate
        decay_steps = config.decay_steps

        # initialize memory and prototypes
        num_class = len(sampler.id2rel)

        memory = collections.defaultdict(list)
        cur_all_acc = []
        his_all_acc = []
        test_cur = []
        test_total = []
        test_top_cur = []
        test_top_total = []
        if verify_history and fix_labels:
            his_data = {}
        total_class = []
        tr_total_top = []
        tr_total = []
        most_high_his = []
        all_high_his = []
        # load data and start computation
        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(sampler):
            # prepare test data
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]

            print(current_relations)
            total_class += current_relations
            # classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(total_class)).to(config.device)
            cur_rel_id = set(rel2id[i] for i in current_relations)

            temp_protos = []  # for memory network
            # for relation in seen_relations:
            #     if relation not in current_relations:
            #         temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))

            # Initial
            # 注意！！！！！！！数据更新可能不对！！！！！

            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            # for d in memory.values():#加入mem中的训练数据
            #     train_data_for_initial.extend(d)
            # memory_ins = []
            # for  ins_list in memory.values():
            #     for _, ins in ins_list:
            #         memory_ins.append(ins)
            # mem_datas=quads2origin_data(memory_ins, id2sentence)

            # First Training
            if d_pic:
                if steps == 0:
                    tokens_task1 = train_data_for_initial
                if steps == 1:
                    tokens_task2 = train_data_for_initial
            # classification_data = train_data_for_initial # new data
            # warm-up
            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial,
                               config.step1_epochs, current_relations, rel2id, fix_labels)

            # classification_data =list(itertools.chain(*(memory.values())))
            for relation in current_relations:
                rel_id = rel2id[relation]
                memory[rel_id] = select_data(config, encoder, dropout_layer,
                                             training_data[relation])  # 训练以后会选择出个K个数据又记录下来

            rel_rep = {}
            for rel_id, ins_list in memory.items():
                rel_rep[rel_id] = get_proto(config, encoder, dropout_layer, ins_list)

            cont_total_acc, topk_total_acc = evaluate_contrastive_model(config, dropout_layer, encoder,
                                                                        contrastive_network, memory,
                                                                        test_data_2,
                                                                        test_emb=rel_rep,
                                                                        use_mem_net=False,
                                                                        new_rel=cur_rel_id)
            print(f"after simple training:{cont_total_acc}")
            # if len(classification_data):
            #     train_first(config, encoder, dropout_layer, classifier, classification_data, total_class, rel2id,
            #             fix_labels, FUNCODE=FUNCODE, n_rel=cur_rel_id)
            #     cont_total_acc, topk_total_acc = evaluate_contrastive_model(config, dropout_layer, encoder,
            #                                                                 contrastive_network, memory,
            #                                                                 test_data_2,
            #                                                                 use_mem_net=False,
            #                                                                 new_rel=cur_rel_id)
            #     print(f"after first training:{cont_total_acc}")

            if verify_history and fix_labels:
                if len(his_data) > 0:
                    cum_acc = 0
                    for step, datasets in his_data.items():
                        acc = valid_classification(config, encoder, dropout_layer, classifier, datasets, fix_labels)
                        cum_acc += acc
                        print(f"valid past task{step}: {acc}")
                    print(f"cumulative history task performance is {cum_acc / (step + 1)}")

            # picture before contrastive learning
            if d_pic:
                num_points = 50
                if steps == 1:
                    flag = False
                    tsne_plot(config, encoder, dropout_layer, tokens_task1, tokens_task2, flag, num_points)

            # # generate prototype
            # memory=slt_to_mem_by_uncty(config, encoder, dropout_layer, classifier, train_data_for_initial,
            #                     current_relations, rel2id,fix_labels,  n_rel=cur_rel_id)
            # memory=trans_memory_format    (memory,cur_rel_id)#后面没有同样数据了，所以这么做OK的

            # # uniform prototype
            # config.fix_sent=True
            # ctst_dload = sample_dataloader(quadruple=None, memory=memory, rel_rep=None,
            #                                id2sent=None,
            #                                config=config,
            #                                seed=config.seed + steps * 100, FUN_CODE=4,
            #                                task_sample=False)
            # optimizer = optim.Adam([
            #     {'params': contrastive_network.parameters(), 'lr': 4e-5},
            # ])
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size=decay_steps,
            #                                             gamma=decay_rate)
            # inp_dict = {
            #     "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #     "scheduler": scheduler,  "dataloader": ctst_dload,
            #     "evaluator": None,
            #     "epoch": config.step2_epochs,
            # }
            # train_contrastive(**inp_dict)

            # for relation in current_relations:
            #     memory[rel2id[relation]] = select_data(config, encoder, dropout_layer,
            #                                            training_data[relation])  # 训练以后会选择出个K个数据又记录下来

            # # uniform prototype
            # task_sample = False
            # ctst_dload = sample_dataloader(quadruple=train_data_for_initial, rel_rep=rel_rep, memory=memory,
            #                                id2sent=id2sentence,
            #                                config=config,
            #                                seed=config.seed + steps * 100, FUN_CODE=3,
            #                                task_sample=task_sample)
            # if use_mem_network:
            #     optimizer = optim.Adam([
            #         {'params': contrastive_network.parameters(), 'lr': 4e-5},
            #         {'params': memory_network.parameters(), 'lr': 1e-4}
            #     ])
            #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                                 step_size=decay_steps,
            #                                                 gamma=decay_rate)
            #     inp_dict = {
            #         "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #         "scheduler": scheduler,  "dataloader": ctst_dload, "evaluator": None,
            #         "epoch": config.step3_epochs, "memory_network": memory_network, "mem_data": temp_protos,
            #         "FUNCODE": FUNCODE,
            #     }
            # else:
            #     optimizer = optim.Adam([
            #         {'params': contrastive_network.parameters(), 'lr': 4e-5},
            #
            #     ])
            #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                                 step_size=decay_steps,
            #                                                 gamma=decay_rate)
            #     inp_dict = {
            #         "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #         "scheduler": scheduler,  "dataloader": ctst_dload, "evaluator": None,
            #         "epoch": config.step3_epochs,
            #     }
            # train_contrastive(**inp_dict)



            # rel_rep_all=[]
            # for rel_id, ins_list in memory.items():
            #     rel_rep_all.append(get_proto(config, encoder, dropout_layer, ins_list))
            #
            # rel_rep_all=torch.cat(rel_rep_all,dim=0).numpy()
            # rel_rep_all = linalg.orth(rel_rep_all.T).T
            #
            # rel_rep = {}
            # for i,rel_id in enumerate(memory.keys()):
            #     rel_rep[rel_id] = torch.unsqueeze(torch.tensor(rel_rep_all[i],device=config.device),dim=0)

            # rel_rep = {}
            # for rel_id, ins_list in memory.items():
            #     rel_rep[rel_id] = get_proto(config, encoder, dropout_layer, ins_list)

            # Second training
            # if steps==0:
            #     config.step2_epochs=int(config.step2_epochs*3)
            #     config.step3_epochs=int(config.step3_epochs*3)
            # elif steps==1:
            #     config.step2_epochs = config.step2_epochs //3
            #     config.step3_epochs = config.step3_epochs //3
            config.fix_sent = False
            config.CDE = 0
            task_sample = True
            use_mem_data = True
            ctst_dload = sample_dataloader(quadruple=train_data_for_initial, memory=memory, rel_rep=rel_rep,
                                           id2sent=id2sentence,
                                           config=config,
                                           seed=config.seed + steps * 100, FUN_CODE=3,
                                           task_sample=task_sample, use_mem_data=use_mem_data)

            # memory = select_to_memory(config, encoder, dropout_layer, classifier, train_data_for_initial,
            #                           memory)  # 须在采样之后，否则本轮中会有memory

            if use_mem_network:
                memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
                                                             input_size=encoder.output_size,
                                                             output_size=encoder.output_size,
                                                             key_size=config.key_size,
                                                             head_size=config.head_size
                                                             ).to(config.device)
                for ins_list in memory.values():
                    temp_protos.append(get_proto(config, encoder, dropout_layer, ins_list))
                temp_protos = torch.cat(temp_protos, dim=0).detach()  # 新和老关系都被选择到了
                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 4e-5},
                    {'params': memory_network.parameters(), 'lr': 1e-4},
                    {'params': encoder.parameters(), 'lr': 1e-4},
                    {'params': dropout_layer.parameters(), 'lr': 1e-4},
                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "dataloader": ctst_dload,
                    "evaluator": None,
                    "epoch": config.step2_epochs, "memory_network": memory_network, "mem_data": temp_protos,
                    "FUNCODE": FUNCODE,
                    "encoder": encoder,
                    "dropout_layer": dropout_layer,
                }
            else:
                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 8e-6},
                    {'params': encoder.parameters(), 'lr': 4e-5},
                    {'params': dropout_layer.parameters(), 'lr': 6e-5},
                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "dataloader": ctst_dload,
                    "evaluator": None,
                    "epoch": config.step2_epochs,
                    "encoder": encoder,
                    "dropout_layer": dropout_layer,
                    "n_rel": cur_rel_id,

                }

            # inp_dict = {
            #     "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #     "scheduler": scheduler,  "dataloader": ctst_dload, "evaluator": None,"FUNCODE":1,
            # }

            train_contrastive(**inp_dict)

            # Third training
            # for relation in current_relations:
            #     memory[rel2id[relation]] = select_data(config, encoder, dropout_layer,
            #                                            training_data[relation])  # 训练以后会选择出个K个数据又记录下来
            # temp_protos = []
            # for ins_list in memory.values():
            #     temp_protos.append(get_proto(config, encoder, dropout_layer, ins_list))
            # temp_protos = torch.cat(temp_protos, dim=0).detach()  # 新和老关系都被选择到了

            # torch.cuda.empty_cache()
            # task_sample = False
            # config.CDE = 2
            # ctst_dload = sample_dataloader(quadruple=train_data_for_initial, rel_rep=rel_rep, memory=memory,
            #                                id2sent=id2sentence,
            #                                config=config,
            #                                seed=config.seed + steps * 100, FUN_CODE=3,
            #                                task_sample=task_sample)
            # if use_mem_network:
            #     optimizer = optim.Adam([
            #         {'params': contrastive_network.parameters(), 'lr': 1e-5},
            #         {'params': memory_network.parameters(), 'lr': 1e-4},
            #         {'params': encoder.parameters(), 'lr': 4e-4},
            #         {'params': dropout_layer.parameters(), 'lr': 6e-4},
            #     ])
            #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                                 step_size=decay_steps,
            #                                                 gamma=decay_rate)
            #     inp_dict = {
            #         "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #         "scheduler": scheduler,  "dataloader": ctst_dload, "evaluator": None,
            #         "epoch": config.step3_epochs, "memory_network": memory_network, "mem_data": temp_protos,
            #         "FUNCODE": FUNCODE,
            #         "encoder": encoder,
            #         "dropout_layer": dropout_layer,"n_rel":cur_rel_id,
            #     }
            # else:
            #     optimizer = optim.Adam([
            #         {'params': contrastive_network.parameters(), 'lr': 1e-6},
            #         {'params': encoder.parameters(), 'lr': 4e-5},
            #         {'params': dropout_layer.parameters(), 'lr': 8e-5},
            #     ])
            #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                                 step_size=decay_steps,
            #                                                 gamma=decay_rate)
            #     inp_dict = {
            #         "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
            #         "scheduler": scheduler,  "dataloader": ctst_dload, "evaluator": None,
            #         "epoch": config.step3_epochs,
            #         "encoder": encoder,
            #         "dropout_layer": dropout_layer,
            #         "n_rel":cur_rel_id,
            #     }
            # train_contrastive(**inp_dict)

            # # WA
            # contrastive_network.align_norms()
            # train_simple_model(config, encoder, dropout_layer, classifier, list(itertools.chain(*(memory.values()))),
            #                    config.step1_epochs, total_class, rel2id, fix_labels)

            if verify_history and fix_labels:
                his_data[steps] = train_data_for_initial
            # picture after contrastive learning
            if d_pic:
                if steps == 1:
                    flag = True
                    tsne_plot(config, encoder, dropout_layer, tokens_task1, tokens_task2, flag, num_points)


            # trash clean
            ctst_dload = None
            # cur_acc = evaluate_first_model(config, encoder, dropout_layer, classifier, test_data_1, seen_relations)
            # total_acc = evaluate_first_model(config, encoder, dropout_layer, classifier, test_data_2, seen_relations)
            # cur_all_acc.append(cur_acc)
            # his_all_acc.append(total_acc)
            # print(f'\nFirst model current test acc:{cur_all_acc}')
            # print(f'First model history test acc:{his_all_acc}')

            # protos4eval = {}
            # for label_id,ins_list in memory.items():
            #     protos4eval[label_id]=get_proto(config, encoder, dropout_layer, ins_list)
            # # protos4eval = torch.cat(protos4eval, dim=0).detach()  # 新和老关系都被选择到了
            # protos4eval = []
            # for label_id,ins_list in memory.items():
            #     protos4eval.append(get_proto(config, encoder, dropout_layer, ins_list))
            # protos4eval = torch.cat(protos4eval, dim=0).detach()  # 新和老关系都被选择到了
            # current evaluation
            if not use_mem_network:
                memory_network = None
                protos4eval = None
            cont_cur_acc, topk_cur_acc = evaluate_contrastive_model(config, dropout_layer, encoder, contrastive_network,
                                                                    memory, test_data_1,
                                                                    memory_network=memory_network,
                                                                    protos4eval=protos4eval,
                                                                    use_mem_net=use_mem_network,
                                                                    new_rel=cur_rel_id)

            cont_total_acc, topk_total_acc = 0, 0

            cont_total_acc, topk_total_acc = evaluate_contrastive_model(config, dropout_layer, encoder,
                                                                        contrastive_network, memory,
                                                                        test_data_2,
                                                                        memory_network=memory_network,
                                                                        protos4eval=protos4eval,
                                                                        use_mem_net=use_mem_network,
                                                                        new_rel=cur_rel_id)

            # for i in memory.values():
            #     random.shuffle(i)

            print(f'\nContrastive model current test acc:{cont_cur_acc}')
            print(f'Contrastive model history test acc:{cont_total_acc}')
            test_cur.append(cont_cur_acc)
            test_total.append(cont_total_acc)

            # test_top_cur.append(topk_cur_acc)
            # test_top_total.append(topk_total_acc)
            # print(f'\nContrastive model current topK test acc:{topk_cur_acc}')
            # print(f'Contrastive model history topK acc:{topk_total_acc}')
            print("\ncontrastive all:")
            print(test_cur)
            print(test_total)
            # print("\ncontrastive topk all:")
            # print(test_top_cur)
            # print(test_top_total)
            a, b = evaluate_contrastive_model(config, dropout_layer, encoder, contrastive_network, memory, test_data_2,
                                              memory_network=memory_network,
                                              protos4eval=protos4eval, use_mem_net=use_mem_network, test_emb=rel_rep,
                                              new_rel=cur_rel_id)
            tr_total.append(a)
            tr_total_top.append(b)
            print(
                f"Contrastive model use training embedding to test: history test acc:{tr_total},Contrastive model history topK acc:{tr_total_top}")
            most_high_his.append(max(a, cont_total_acc))
            print(
                f"highest history test acc:{most_high_his}")
            if steps == len(sampler) - 1:
                all_results_cur.append(test_cur)
                all_results.append(test_total)
                all_high_his.append(most_high_his)

    print(f"all current results:{all_results_cur}")
    print(f"all history results:{all_results}")

    print(f"current average:{np.average(np.array(all_results_cur), axis=0)}")
    print(f"history average:{np.average(np.array(all_results), axis=0)}")

    print(f"highest history average:{np.average(np.array(all_high_his), axis=0)}")

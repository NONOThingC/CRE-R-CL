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
import time

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from config import Config
import torch.nn.functional as F

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.dropout_layer import Dropout_Layer
from model.classifier.softmax_classifier import Softmax_Layer
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified

from utils import outputer, batch2device

from Sampler.sample_dataloader import sample_dataloader, data_sampler
from data_loader import get_data_loader


def train_contrastive(config, logger, model, optimizer, scheduler, loss_func, dataloader, evaluator):
    train_dataloader, valid_dataloader = dataloader

    for ep in range(config.train_epoch):
        ## train
        model.train()
        t_ep = time.time()
        # epoch parameter start
        start_lr = optimizer.param_groups[0]['lr']
        batch_cum_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        # epoch parameter end
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            # batch parameter start
            t_batch = time.time()

            # batch parameter end

            # move to device
            batch_train_data = batch2device(batch_train_data, config.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            sent_inp, emb_inp, labels, comparison = batch_train_data
            # model forward start
            inp_dict = {

            }
            # or
            inp_lst = [sent_inp, emb_inp, comparison]

            m_out = model(*inp_lst)
            # model forward end

            # grad operation start
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = m_out

            w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
            loss = w_ent * loss_func(
                ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func(
                head_rel_shaking_outputs,
                batch_head_rel_shaking_tag) + w_rel * loss_func(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss.backward()
            optimizer.step()

            # grad operation end

            # accuracy calculation start
            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item()

            batch_cum_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            batch_avg_loss = batch_cum_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind +
                                                                   1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind +
                                                                   1)
            # accuracy calculation end
            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                experiment_name,
                config["run_name"],
                ep + 1,
                config.train_epoch,
                batch_ind + 1,
                len(train_dataloader),
                batch_avg_loss,
                avg_ent_sample_acc,
                avg_head_rel_sample_acc,
                avg_tail_rel_sample_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")
            # batch logger and print end

            # change lr
            scheduler.step()
        # epoch logger and print start
        logger.log({
            "train_loss": batch_avg_loss,
            "train_ent_seq_acc": avg_ent_sample_acc,
            "train_head_rel_acc": avg_head_rel_sample_acc,
            "train_tail_rel_acc": avg_tail_rel_sample_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "time": time.time() - t_ep,
        })
        # epoch logger and print start

    # epoch logger and print start
    avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
    avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
    avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

    log_dict = {
        "val_ent_seq_acc": avg_ent_sample_acc,
        "val_head_rel_acc": avg_head_rel_sample_acc,
        "val_tail_rel_acc": avg_tail_rel_sample_acc,
        "valid epoch time": time.time() - t_ep,
    }
    logger.log(log_dict)
    RS_logger.ts_log().add_scalars("Teacher Valid", {
        "e_acc": avg_ent_sample_acc,
        "h_acc": avg_head_rel_sample_acc,
        "t_acc": avg_tail_rel_sample_acc,
    }, RS_logger.get_cur_ep())
    # epoch logger and print end
    return (avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)


def contrastive_loss(hidden, labels, temperature=0.4,
                     weights=1.0):
    LARGE_NUM = 1e9
    # hidden=torch.linalg.norm(hidden,dim=-1)
    # hidden1,hidden2=torch.split(hidden, 2, dim=0)
    # batch_size = hidden1.shape[0]
    # hidden1_large = hidden1
    # hidden2_large = hidden2
    CEL = nn.CrossEntropyLoss(weight=weights)
    # labels = torch.one_hot(torch.range(batch_size), batch_size * 2)#因为是和自己做所以只有对角线为正
    # masks = torch.one_hot(torch.range(batch_size), batch_size)#注意这个函数pytorch无

    # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,-1,-2)) / temperature
    # logits_aa = logits_aa - masks * LARGE_NUM#这个是为什么？
    # logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large,-1,-2)) / temperature
    # logits_bb = logits_bb - masks * LARGE_NUM
    # logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,-1,-2)) / temperature
    # logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,-1,-2)) / temperature

    loss_a = CEL(
        torch.concat([logits_ab, logits_aa], 1), labels, weights=weights)
    # loss_a = CEL(
    #     torch.concat([logits_ab, logits_aa], 1),labels,  weights=weights)
    # loss_b = CEL(
    #      torch.concat([logits_ba, logits_bb], 1),labels,  weights=weights)
    loss = loss_a + loss_b

    return loss


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


# def computer_CrossEntropyLoss(predicts,labels):

def train_first(config, encoder, dropout_layer, classifier, training_data, epochs):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    id2sent = []
    ret_d = []
    for epoch_i in range(epochs):
        losses = []
        if epoch_i == 0:
            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, 2)
        for step, (labels, tokens) in enumerate(data_loader):
            # with torch.no_grad():
            optimizer.zero_grad()
            out_prob = []
            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
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
            out_std = torch.std(out_prob, dim=0)
            # out_std_all = []
            # for i in range(config.f_pass):
            #     out_std_all.append(out_std)#xx
            # out_std_all = torch.stack(out_std_all)
            out_std_all = out_std.expand((config.f_pass, out_std.shape[0], -1))  # mBC
            # uncertainty
            out_std_mask = out_std < config.kappa_pos  # 不满足被标志为1之后去除 #BC

            out_std_mask = out_std_mask.expand((config.f_pass, out_std_mask.shape[0], -1))  # m B C
            max_idx = torch.argmax(out_prob, dim=-1)  # m,B
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            labels_mask = torch.zeros_like(out_std_mask).scatter_(-1, m_labels.view(m_labels.shape[0], -1, 1),
                                                                  1)  # m,B,C
            preds_mask = torch.zeros_like(out_std_mask).scatter_(-1, max_idx.view(max_idx.shape[0], -1, 1), 1)  # m,B,C

            p_mask = (labels_mask == preds_mask) * out_std_mask  # m,B,C
            n_mask = (labels_mask != preds_mask) * out_std_mask  # m,B,C
            # torch.index_select(x, 0, indices)
            p_index = (p_mask.sum(dim=-1) > 0)  # m B
            n_index = (n_mask.sum(dim=-1) > 0)  # m B

            p_labels, n_labels = m_labels[p_index], m_labels[n_index]
            p_logits, n_logits = logits_all[p_index], logits_all[n_index]
            slt_labels = torch.cat([p_labels, n_labels], dim=0)
            slt_logits = torch.cat([p_logits, n_logits], dim=0)

            # slt_labels=m_labels[(out_std_mask.sum(dim=-1)>0)] #对吗？

            # batch_p = torch.index_select(m_labels, 0, p_index)    # B
            # batch_n=(n_mask.sum(dim=-1) > 0).nonzero()
            # logits_all[out_std_mask]=-float('inf')
            # batch_p = torch.index_select(m_labels, 0, p_index)

            # labels=labels*out_std_mask
            loss1 = criterion(slt_logits, slt_labels)
            # logits_all.requires_grad_()
            # for i in range(config.f_pass):
            #     loss1 += criterion(logits_all[i][out_std_mask], labels)
            loss2 = compute_jsd_loss(slt_logits)  # ??
            loss = loss1 + loss2
            loss.backward()
            losses.append(loss.item())
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()

            # data store
            if epoch_i == epochs - 1:

                with torch.no_grad():
                    id2sent.extend(tokens.cpu())
                    # index range:[step * data_loader.batch_size + id,step * data_loader.batch_size + len(tokens))

                    m_tokens_ids = torch.Tensor(
                        range(step * data_loader.batch_size, step * data_loader.batch_size + len(tokens))).expand(
                        config.f_pass, -1)  # m B
                    m_embeddings = torch.stack(output_embeddings)  # m B H
                    slt_tokens_ids = torch.cat([m_tokens_ids[p_index], m_tokens_ids[n_index]], dim=0).int().tolist()
                    slt_embeddings = torch.cat([m_embeddings[p_index], m_embeddings[n_index]], dim=0).cpu()
                    slt_preds = torch.cat([max_idx[p_index], max_idx[n_index]], dim=0).tolist()
                    slt_labels = slt_labels.tolist()
                    for i in range(len(slt_tokens_ids)):
                        ret_d.append((slt_tokens_ids[i], slt_embeddings[i], slt_preds[i], slt_labels[i]))

        print(f"loss is {np.array(losses).mean()}")

    # with torch.no_grad():
    #     for step, (labels, tokens) in enumerate(data_loader):
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

    return id2sent, ret_d


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
    proto = torch.mean(features, dim=0, keepdim=True)  # proto是encoder之后的向量

    # return the averaged prototype
    return proto


# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(config, encoder, sample_set):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []

    for step, (labels, tokens) in enumerate(data_loader):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
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
        mem_set.append(instance)  # 保存的是句子不是embedding
    return mem_set


def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])

    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens) in enumerate(data_loader):
            optimizer.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)
            reps, output1 = dropout_layer(reps)
            logits = classifier(reps)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


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
    for step, (labels, tokens) in enumerate(data_loader):
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


def evaluate_no_mem_model(config, encoder, classifier, test_data):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, (labels, tokens) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        neg_index = random.sample(range(0, 80), 10)
        neg_sim = logits[:, neg_index].cpu().data.numpy()
        max_smi = np.max(neg_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct / n


def evaluate_strict_model(config, encoder, classifier, memory_network, test_data, protos4eval, seen_relations):
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
    for step, (labels, tokens) in enumerate(data_loader):
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


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Config for lifelong relation extraction (classification)")
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    # output result
    printer = outputer()
    middle_printer = outputer()
    start_printer = outputer()

    # set training batch
    for i in range(config.total_round):

        test_cur = []
        test_total = []

        # set random seed
        random.seed(config.seed + i * 100)

        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        # encoder setup
        encoder = Bert_Encoder(config=config).to(config.device)
        # dropout setup
        dropout_layer = Dropout_Layer(config=config, input_size=encoder.output_size).to(config.device)
        # classifier setup
        classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation).to(config.device)

        # record testing results
        sequence_results = []
        result_whole_test = []

        # initialize memory and prototypes
        num_class = len(sampler.id2rel)
        memorized_samples = {}

        id2sentence = []
        quads = []
        # load data and start computation
        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):

            print(current_relations)

            temp_mem = {}
            temp_protos = []
            # for relation in seen_relations:
            #     if relation not in current_relations:
            #         temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))

            # Initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            # first model
            id2sentence_1, quads_1 = train_first(config, encoder, dropout_layer, classifier, train_data_for_initial,
                                                 config.step1_epochs)
            id2sentence.extend(id2sentence_1)
            quads.extend(quads_1)

        with open('id2sentence.pkl', 'wb') as f:
            pickle.dump(id2sentence, f)
        with open('quads.pkl', 'wb') as f:
            pickle.dump(quads, f)

        ctst_dload = sample_dataloader(quadruple=quads, id2sent=id2sentence, config=config, seed=config.seed + i * 100)

        train_contrastive(config)

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
import torch.nn.functional as F

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.dropout_layer import Dropout_Layer
from model.classifier.softmax_classifier import Softmax_Layer
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified

from utils import outputer

from sampler import data_sampler
from data_loader import get_data_loader


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


def id_sentence(config, training_data):
    data_loader = get_data_loader(config, training_data)
    id2sentence = {}
    for step, (labels, tokens) in enumerate(data_loader):
        for id in range(len(tokens)):
            id2sentence[step*data_loader.batch_size+id] = tokens[id].detach().numpy().tolist()
    return id2sentence


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
    for epoch_i in range(epochs):
        losses = []
        if epoch_i == 0:
            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, 2)
        for step, (labels, tokens) in enumerate(data_loader):
            # with torch.no_grad():

            out_prob = []
            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)
            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))
            out_prob = torch.stack(out_prob)
            logits_all = torch.stack(logits_all)
            out_std = torch.std(out_prob, dim=0)
            out_std_all = []
            for i in range(config.f_pass):
                out_std_all.append(out_std)
            out_std_all = torch.stack(out_std_all)
            out_std_mask = out_std_all < config.kappa_pos
            max_value, max_idx = torch.max(out_prob, dim=-1)
            
            '''
            max_std = out_std.gather(1, max_idx.view(-1, 1))
            max_idx=max_idx.cpu()
            max_std=max_std.cpu()
            pos_logits,neg_logits,selected_logits=[],[],[]
            for i in range(config.f_pass):
                neg_idx=((max_idx[i]!=labels)*(max_std.squeeze(1)<config.kappa_neg))
                pos_idx=((max_idx[i]==labels)*(max_std.squeeze(1)<config.kappa_pos))
                pos=logits_all[i][pos_idx]
                neg=logits_all[i][neg_idx]
                pos_logits.append(pos)
                neg_logits.append(neg)
                selected_logits.append(torch.vstack([pos,neg]))
            '''
            
            # logits_all[out_std_mask]=-float('inf')
            labels = labels.to(config.device)
            loss1 = 0
            # logits_all.requires_grad_()
            for i in range(config.f_pass):
                loss1 += criterion(logits_all[i], labels)
            loss2 = compute_jsd_loss(logits_all)
            loss = loss1 + loss2
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")

    for step, (labels, tokens) in enumerate(data_loader):
        id2sentence = {}
        result={}
        for id in range(len(tokens)):
            id2sentence[step * data_loader.batch_size + id] = tokens[id].detach().numpy().tolist()
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        for _ in range(config.f_pass):
            output, output_embedding = dropout_layer(reps)
            logits = classifier(output)
            out_prob=F.softmax(logits, dim=-1)
            max_value, max_idx = torch.max(out_prob, dim=-1)
            output_embedding,max_idx=output_embedding.cpu(),max_idx.cpu()
            output_embedding=output_embedding.detach().numpy().tolist()
            max_idx=max_idx.detach().numpy().tolist()
            labels=labels.detach().numpy().tolist()
            for id in range(len(labels)):
                mid=[step * data_loader.batch_size + id, output_embedding[id], max_idx[id],labels[id]]
                if labels[id] not in result:
                    result[labels[id]]=[mid]
                else:
                    result[labels[id]].append(mid)

        with open('id2sentence.json', 'w+') as f:
            json.dump(id2sentence, f)
            
    return id2sentence,result


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
            encoder.zero_grad()
            dropout_layer.zero_grad()
            classifier.zero_grad()

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

        id2sentence={}
        # load data and start computation
        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):

            print(current_relations)

            temp_mem = {}
            temp_protos = []
            for relation in seen_relations:
                if relation not in current_relations:
                    temp_protos.append(get_proto(config, encoder, memorized_samples[relation]))

            # Initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            # train model
            # train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step1_epochs//3)

            # id_sentence
            id2sentence_1=id_sentence(config, train_data_for_initial)
            id2sentence.update(id2sentence_1)
            # first model
            train_first(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step1_epochs)

            # # Memory Activation
            # train_data_for_replay = []
            # random.seed(config.seed+i*100)
            # for relation in current_relations:
            #     train_data_for_replay += training_data[relation]
            # for relation in memorized_samples:
            #     train_data_for_replay += memorized_samples[relation]
            # train_simple_model(config, encoder, classifier, train_data_for_replay, config.step2_epochs)

            for relation in current_relations:
                temp_mem[relation] = select_data(config, encoder, training_data[relation])
                temp_protos.append(get_proto(config, encoder, temp_mem[relation]))
            temp_protos = torch.cat(temp_protos, dim=0).detach()  # detach：设置required_grad为False，不再改变proto的值

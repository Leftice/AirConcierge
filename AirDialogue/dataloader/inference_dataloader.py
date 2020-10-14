import os
import torch
import json

import codecs
import collections
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import time
import matplotlib.pyplot as plt
import random

UNK = "<unk>"
UNK_ID = 0

start_of_turn1 = "<t1>"
start_of_turn2 = "<t2>"
end_of_dialogue = "<eod>"

global eod_id
global smallkb_n
global empty_id

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.size = 0
        self.kb_size = 0

    def create_vocab_table(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for line in f:
                words = line.split()
                for word in words:
                    _ = self.dictionary.add_word(word)

    def infer_tokenize(self, path_src, path_tar, mask=False):
        sents = []
        tar_sents = []
        sents_len = [] 
        count = 0
        true_answer = []

        tar_fp = open(path_tar, "r")
        with open(path_src, 'r') as f:
            for line in f:
                tar_line = tar_fp.readline()
                src_items = line.split("|")
                tar_items = tar_line.split("|")
                sent = []
                tar_sent = []

                # intent and goal
                intent_item = src_items[0].split()
                intent_goal = intent_item[14].split('_', 1)[1].split('>', 1)[0]
                ground_truth = tar_items[0].split()
                if len(ground_truth) != 4:
                    final_state = ground_truth[1].split('_', 1)[1].split('>', 1)[0]
                else:
                    final_state = ground_truth[3].split('_', 1)[1].split('>', 1)[0]
   
                words = []
                for word in tar_items[1].split(): # since the last one is a word '<t1\n>', above is a int '6\n' ->int() -> '6'
                    # mask word flight number 
                    if mask and ((intent_goal == 'book' and final_state == 'book') or (intent_goal == 'change' and final_state == 'change')):
                        if word.isdigit() and (int(word) >= 1001 and int(word) <= 1029):
                            # word = '<mask_flight>'
                            word = '<fl_' + str(word) + '>'
                            count += 1
                        else:
                            for f in range(1001, 1030):
                                str_f = str(f)
                                if str_f in word:
                                    # word = '<mask_flight>'
                                    word = '<fl_' + str_f + '>'
                                    count += 1
                        if '1000' in word and 'ight' in words[-4:]:
                            # word = '<mask_flight>'
                            word = '<fl_1000>'
                            count += 1
                    
                    try:
                        words.append(self.dictionary.word2idx[word])
                    except KeyError:
                        words.append(self.dictionary.word2idx[UNK])
                tar_sents.append(words)

                for word in tar_items[0].split(" "):# true kb answer
                    if '<fl' in word:
                        token = word.split('_', 1)[1].split('>', 1)[0]
                        if token == 'empty':
                            word = 30
                            true_answer.append(int(word))
                        if token.isdigit() and (int(token) >= 1000 and int(token) <= 1029):
                            word = int(token) - 1000 # 1000~1029 -> 0 ~ 29
                            true_answer.append(int(word))

                for i in range(2):
                    words = []
                    for word in src_items[i].split():

                        # mask word flight number 
                        if mask and i == 1 and ((intent_goal == 'book' and final_state == 'book') or (intent_goal == 'change' and final_state == 'change')):
                            if word.isdigit() and (int(word) >= 1001 and int(word) <= 1029):
                                # word = '<mask_flight>'
                                word = '<fl_' + str(word) + '>'
                                count += 1
                            else:
                                for f in range(1001, 1030):
                                    str_f = str(f)
                                    if str_f in word:
                                        # word = '<mask_flight>'
                                        word = '<fl_' + str_f + '>'
                                        count += 1
                            if '1000' in word and 'ight' in words[-4:]:
                                # word = '<mask_flight>'
                                word = '<fl_1000>'
                                count += 1

                        if i == 0 or i==1 : # tokenize intent, dialogue
                            try:
                                words.append(self.dictionary.word2idx[word])
                            except KeyError:
                                words.append(self.dictionary.word2idx[UNK])
                                if i == 0:
                                    print('Error token in infer : ', word)
                                    print('words : ', words)
                                    raise
                    sent.append(words)
                sents.append(sent)
                sents_len.append(len(sent[1]))
                self.size += 1
            self.data = sents
        print('Count : ', 100. * count/len(sents))
        tar_fp.close()
        return sents, sents_len, tar_sents, true_answer

    def tokenize_kb(self, path):
        empty_id = 3
        # <res_no_res> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_21> <cl_business> <pr_800> <cn_1> <al_AA> <fl_1000> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_21> <tn2_0> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1001> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_21> <tn2_6> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1002> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_21> <tn2_2> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1003> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_13> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1004> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_1> <tn2_15> <cl_economy> <pr_100> <cn_0> <al_Frontier> <fl_1005> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_12> <tn1_8> <tn2_21> <cl_economy> <pr_200> <cn_1> <al_Delta> <fl_1006> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_6> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1007> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_23> <tn2_12> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1008> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_21> <tn2_14> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1009> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_12> <cl_business> <pr_500> <cn_1> <al_Southwest> <fl_1010> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1011> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_13> <d2_12> <tn1_0> <tn2_21> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1012> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_7> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1013> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_7> <tn2_0> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1014> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1015> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_13> <tn1_23> <tn2_18> <cl_economy> <pr_200> <cn_1> <al_Hawaiian> <fl_1016> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_3> <tn2_17> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1017> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_8> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1018> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_17> <tn2_14> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1019> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_20> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1020> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_5> <tn2_15> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1021> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_12> <tn1_12> <tn2_5> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1022> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_16> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1023> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_7> <cl_economy> <pr_100> <cn_1> <al_Spirit> <fl_1024> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_11> <tn2_16> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1025> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_8> <tn2_1> <cl_economy> <pr_100> <cn_1> <al_Hawaiian> <fl_1026> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1027> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_17> <tn2_23> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1028> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1029>
        kb_sents = []
        with open(path, 'r') as f:
            for line in f:
                words = []
                if len(line.split()) > 13:
                    for word in line.split():
                        try:
                            words.append(self.dictionary.word2idx[word])
                        except KeyError:
                            words.append(self.dictionary.word2idx[UNK])
                            print('tokenize_kb : ', word)
                            raise
                else:
                    words.append(self.dictionary.word2idx[line.split()[0]]) # empty & reservation
                for i in range(13):
                    words.append(empty_id)
                kb_sents.append(words)
                self.kb_size += 1
            self.kb_data = kb_sents
        return kb_sents

    def tokenize_column(self, sql_paths, table_paths):
        table_data = {}
        sql_data = []
        max_col_num = 0

        if sql_paths is not None:
            print ('Loading data from {}'.format(sql_paths))
            with open(sql_paths) as inf:
                for line in inf:
                    sql = json.loads(line.strip())
                    sql_data.append(sql)

        print ('Loading data from {}'.format(table_paths))
        with open(table_paths) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                # print('header_tok : ', len(tab[u'header_tok']))
                column_name = []
                column_len = []
                for name in tab[u'header_tok']:
                    words = []
                    column_len.append(len(name))
                    for word in name:
                        try:
                            words.append(self.dictionary.word2idx[word])
                        except KeyError:
                            words.append(self.dictionary.word2idx[UNK])
                            print('tokenize_column : ', word)
                            raise
                    column_name.append(words)

                table_data[tab[u'id']] = tab
                table_data[tab[u'id']]['header_tok'] = column_name
                table_data[tab[u'id']]['header_len'] = column_len

        return sql_data, table_data

def read_fileter_kb(path):
    with open(path, 'r') as f:
        all_filter_index = []
        all_turn_gate = []
        for line in f:
            items = line.split("|")
            filter_index = []
            turn_gate = []
            for word in items[0].split():
                filter_index.append(int(word))
            for word in items[1].split():
                turn_gate.append(int(word))
            filter_index.append(1) # empty flight
            all_filter_index.append(filter_index)
            all_turn_gate.append(turn_gate)
            if len(filter_index) != 31:
                print('Error, filter should have 30 !')
                raise
    return all_filter_index, all_turn_gate

def read_turn_gate(path):
    with open(path, 'r') as f:
        all_turn_gate = []
        for line in f:
            line = line.split()
            all_turn_gate.append(int(line[0]))
    return all_turn_gate

def process_entry_inferenced(inputs, vocab_table, t1_id, t2_id, table_data):
    """Pre-process procedure for the supervised iterator."""
    intent, dialogue, kb, tar_dialogue, kb_true_answer, filter_index, turn_gate, turn_gate_index = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]

    size_intent = len(intent)
    size_dialogue = len(dialogue)
    has_reservation = kb[0]
    kb = kb[1:]

    size_dialogue = size_dialogue
    source_diag = dialogue
    # target_diag = [dialogue[-1]] + tar_dialogue[:-1]
    target_diag = [dialogue[-1]] + tar_dialogue

    col_seq = table_data['0-0000']['header_tok']
    col_num = len(table_data['0-0000']['header_tok'])

    return intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer, filter_index, turn_gate, turn_gate_index

class AirData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def list_to_tensor(ll):
    tensor_list = []
    for i in ll:
        tensor_list.append(torch.tensor(np.array(i).astype(int)))
    return tensor_list

def collate_fn_inference(data):
    global eod_id
    global smallkb_n
    data.sort(key=lambda x: len(x[2]), reverse=True)
    intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer, filter_index, turn_gate, turn_gate_index = zip(*data)

    intent_pad = pad_sequence(list_to_tensor(intent), batch_first=True, padding_value=eod_id)
    size_intent = torch.tensor(size_intent, dtype=torch.int64)
    source_diag_pad = pad_sequence(list_to_tensor(source_diag), batch_first=True, padding_value=eod_id)
    target_diag_pad = pad_sequence(list_to_tensor(target_diag), batch_first=True, padding_value=eod_id)
    size_dialogue = torch.tensor(size_dialogue, dtype=torch.int64)

    num_kb = len(kb[0]) // 13
    kb = torch.tensor(kb, dtype=torch.int64).view(-1, num_kb, 13)
    # if num_kb > 2:
    #     kb_shuffle = list(range(num_kb))
    #     random.shuffle(kb_shuffle)
    #     kb = kb[:, kb_shuffle]

    has_reservation = torch.tensor(has_reservation, dtype=torch.int64)
    col_seq = torch.tensor(col_seq) # (b, 12, 1)
    col_num = torch.tensor(col_num, dtype=torch.int64)

    return intent_pad, size_intent, source_diag_pad, target_diag_pad, size_dialogue, kb, has_reservation, col_seq, turn_gate, turn_gate_index

def Inference_loader_2(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30, args=None):
    global eod_id
    global smallkb_n
    smallkb_n = small_n
    # print('Kb_n : ', smallkb_n)

    if args.syn:
        pre_sql_path = './results/synthesized/'
        pre_data_path = './data/synthesized/'
    elif args.air:
        pre_sql_path = './results/airdialogue/'
        pre_data_path = './data/airdialogue/'
    else:
        print('Pleae use --syn or --air !')
        raise

    # test
    vocab_file = pre_data_path + 'tokenized/vocab.txt'
    if toy : 
        src_data_file = pre_data_path + 'tokenized/sub_air/toy_dev.infer.src.data'
        tar_data_file = pre_data_path + 'tokenized/sub_air/toy_dev.infer.tar.data'
        kb_file = pre_sql_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev/dev_tok.tables.jsonl'
        filtered_index_path = pre_sql_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered_index.kb'
        turn_gate_path = pre_sql_path + 'Inference_Bleu/SQL/prior_gate.txt'
        print('Using toy !')
    elif dev:
        src_data_file = pre_data_path + 'tokenized/infer/dev.infer.src.data'
        tar_data_file = pre_data_path + 'tokenized/infer/dev.infer.tar.data'
        kb_file = pre_sql_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev/dev_tok.tables.jsonl'
        filtered_index_path = pre_sql_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered_index.kb'
        turn_gate_path = pre_sql_path + 'Inference_Bleu/SQL/prior_gate.txt'
    else:
        src_data_file = pre_data_path + 'tokenized/infer/dev.infer.src.data'
        tar_data_file = pre_data_path + 'tokenized/infer/dev.infer.tar.data'
        kb_file = pre_data_path + 'tokenized/infer/dev.infer.kb'
        table_path = pre_data_path + 'SQL/dev/dev_tok.tables.jsonl'
        filtered_index_path = pre_sql_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered_index.kb'
        turn_gate_path = pre_sql_path + 'Inference_Bleu/SQL/prior_gate.txt'

    # vocab table & tokenize
    corpus = Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    # print('Use mask : ', mask)
    sents, sents_len, tar_sents, kb_true_answer = corpus.infer_tokenize(src_data_file, tar_data_file, mask)
    print('Reading kb')
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(None, table_path)
    filter_index, turn_gate = read_fileter_kb(filtered_index_path)
    turn_gate_index = read_turn_gate(turn_gate_path)

    print('Building Dataset')
    if n_sample != -1:
        sents = sents[:n_sample]
        sents_len = sents_len[:n_sample]
        kb_sents = kb_sents[:n_sample]
        kb_true_answer = kb_true_answer[:n_sample]

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(tar_sents[i])
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(filter_index[i])
        combined_dataset[i].append(turn_gate[i])
        combined_dataset[i].append(turn_gate_index[i])

    if only_f:
        print('Error , No --only_f in inference mode !')
        raise

    eod_id= corpus.dictionary.word2idx['<eod>']
    t1_id = corpus.dictionary.word2idx['<t1>']
    t2_id = corpus.dictionary.word2idx['<t2>']
    unk = corpus.dictionary.word2idx['<unk>']
    combined_dataset = map(partial(process_entry_inferenced, vocab_table=corpus, t1_id=t1_id, t2_id=t2_id, table_data=table_data), combined_dataset)

    data = AirData(combined_dataset)
    if dev :
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_inference, drop_last=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_inference, drop_last=False)

    return data_loader, corpus
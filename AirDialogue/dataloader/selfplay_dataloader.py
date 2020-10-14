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

    def self_play_eval_tokenize(self, path):
        sents = []
        true_answer = []
        action_state = []
        SQL_YN = []
        with open(path, 'r') as f:
            for line in f:
                items = line.split("|")
                sent = []

                # intent and goal
                intent_item = items[0].split()
                intent_goal = intent_item[14].split('_', 1)[1].split('>', 1)[0]
                ground_truth = items[1].split()
                if len(ground_truth) != 4:
                    final_state = ground_truth[1].split('_', 1)[1].split('>', 1)[0]
                    print('len(ground_truth) != 4 Self play eval')
                    raise
                else:
                    final_state = ground_truth[3].split('_', 1)[1].split('>', 1)[0]
                if ((intent_goal == 'cancel' and final_state == 'cancel') or (intent_goal == 'change' and final_state == 'no_reservation') or (intent_goal == 'cancel' and final_state == 'no_reservation')):
                    SQL_YN.append(0)
                else:
                    SQL_YN.append(1)
                for i in range(2):
                    words = []
                    for word in items[i].split():
                        # action
                        if i == 1:
                            if '<fl' in word:
                                token = word.split('_', 1)[1].split('>', 1)[0]
                                if token == 'empty':
                                    word = 30
                                    true_answer.append(int(word))
                                    action_state.append(-1)
                                    words.append(word)
                                if token.isdigit() and (int(token) >= 1000 and int(token) <= 1029):
                                    word = int(token) - 1000 # 1000~1029 -> 0 ~ 29
                                    true_answer.append(int(word))
                                    action_state.append(int(token))
                                    words.append(word)
                            else:
                                try:
                                    words.append(self.dictionary.word2idx[word])
                                except KeyError:
                                    words.append(self.dictionary.word2idx[UNK])
                        # tokenize intent
                        if i == 0: 
                            try:
                                words.append(self.dictionary.word2idx[word])
                            except KeyError:
                                words.append(self.dictionary.word2idx[UNK])
                                print('Self Play Eval intent unk !')
                                raise
                    sent.append(words)
                sents.append(sent)
                self.size += 1
            self.data = sents
        return sents, true_answer, action_state, SQL_YN

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
        for line in f:
            items = line.split("|")
            filter_index = []
            for word in items[0].split():
                filter_index.append(int(word))
            filter_index.append(1) # empty flight
            all_filter_index.append(filter_index)
            if len(filter_index) != 31:
                print('Error, filter should have 30 !')
                raise
    return all_filter_index

def read_turn_gate(path):
    with open(path, 'r') as f:
        all_turn_gate = []
        for line in f:
            line = line.split()
            all_turn_gate.append(int(line[0]))
    return all_turn_gate

def process_entry_self_play_eval(inputs, vocab_table, t1_id, t2_id, table_data):
    """Pre-process procedure for the supervised iterator."""
    intent, action, kb, kb_true_answer, SQL_YN, sql_data, filter_index, turn_gate  = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]

    size_intent = len(intent)
    has_reservation = kb[0]
    kb = kb[1:]

    sql = sql_data
    ans_seq =(len(sql['sql']['conds']), tuple(x[0] for x in sql['sql']['conds']), tuple(x[1] for x in sql['sql']['conds']), tuple(x[2] for x in sql['sql']['conds'])) # (x[0]) column list , (x[1]) operator list
    col_seq = table_data[sql['table_id']]['header_tok']
    col_num = len(table_data[sql['table_id']]['header_tok'])
    truth_seq = sql['query']

    col_seq = table_data['0-0000']['header_tok']
    col_num = len(table_data['0-0000']['header_tok'])

    return intent, size_intent, action, kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, kb_true_answer, SQL_YN, filter_index, turn_gate

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

def collate_fn_self_play_eval(data):
    global eod_id
    global smallkb_n
    intent, size_intent, action, kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, kb_true_answer, SQL_YN, filter_index, turn_gate = zip(*data)

    intent_pad = pad_sequence(list_to_tensor(intent), batch_first=True, padding_value=eod_id)
    size_intent = torch.tensor(size_intent, dtype=torch.int64)
    action_pad = torch.tensor(action, dtype=torch.int64)
    
    num_kb = len(kb[0]) // 13
    kb = torch.tensor(kb, dtype=torch.int64).view(-1, num_kb, 13)
    if num_kb > 2:
        kb_shuffle = list(range(num_kb))
        random.shuffle(kb_shuffle)
        kb = kb[:, kb_shuffle]
    # kb_true_answer = torch.tensor(kb_true_answer, dtype=torch.int64)
    # kb_true_answer_onehot = torch.zeros(kb_true_answer.size(0), 31).scatter_(1, kb_true_answer.view(kb_true_answer.size(0), 1).data, 1)

    # filter_index_tensor = pad_sequence(list_to_tensor(filter_index), batch_first=True, padding_value=0) # (b, 30)
    # filter_index_tensor_expand = filter_index_tensor.unsqueeze(2).expand(-1, -1, 13)
    # filter_kb = filter_index_tensor_expand * kb

    has_reservation = torch.tensor(has_reservation, dtype=torch.int64)
    SQL_YN_tensor = torch.tensor(SQL_YN, dtype=torch.int64)
    col_seq = torch.tensor(col_seq) # (b, 12, 1)

    # return intent_pad, size_intent, action_pad, filter_kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, SQL_YN_tensor, kb_true_answer_onehot, kb_true_answer 
    return intent_pad, size_intent, action_pad, kb, has_reservation, col_seq, truth_seq, SQL_YN_tensor, turn_gate

def SelfPlayEval_loader_2(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30, args=None):
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
        src_data_file = pre_data_path + 'tokenized/sub_air/toy_dev.selfplay.eval.data'
        kb_file = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'
        filtered_index_path = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered_index.kb'
        turn_gate_path = pre_sql_path + 'SelfPlay_Eval/SQL/prior_gate.txt'
    else:
        src_data_file = pre_data_path + 'tokenized/selfplay_eval/dev.selfplay.eval.data'
        kb_file = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'
        filtered_index_path = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered_index.kb'
        turn_gate_path = pre_sql_path + 'SelfPlay_Eval/SQL/prior_gate.txt'

    print('SelfPlayEval_loader Loading data : ', src_data_file)
    print('SelfPlayEval_loader Loading kb : ', kb_file)

    # vocab table & tokenize
    corpus = Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    sents, kb_true_answer, action_state, SQL_YN = corpus.self_play_eval_tokenize(src_data_file)
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(sql_path, table_path)
    filter_index = read_fileter_kb(filtered_index_path)
    turn_gate = read_turn_gate(turn_gate_path)

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(SQL_YN[i])
        combined_dataset[i].append(sql_data[i])
        combined_dataset[i].append(filter_index[i])
        combined_dataset[i].append(turn_gate[i])

    if only_f:
        combined_dataset_new = []
        for i in range(len(sents)):
            if action_state[i] != -1 and SQL_YN[i] == 1:
                combined_dataset_new.append(combined_dataset[i])
        print('combined_dataset new : ', len(combined_dataset_new), 100. * len(combined_dataset_new) / len(combined_dataset))
        combined_dataset = combined_dataset_new

    eod_id= corpus.dictionary.word2idx['<eod>']
    t1_id = corpus.dictionary.word2idx['<t1>']
    t2_id = corpus.dictionary.word2idx['<t2>']
    unk = corpus.dictionary.word2idx['<unk>']
    combined_dataset = map(partial(process_entry_self_play_eval, vocab_table=corpus, t1_id=t1_id, t2_id=t2_id, table_data=table_data), combined_dataset)

    data = AirData(combined_dataset)
    if dev :
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_self_play_eval, drop_last=False)

    return data_loader, corpus

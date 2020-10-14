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

    def tokenize(self, path, mask=False):
        sents = []
        sents_len = [] 
        count = 0
        true_answer = []
        action_state = []
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
                else:
                    final_state = ground_truth[3].split('_', 1)[1].split('>', 1)[0]

                for i in range(4):
                    words = []
                    for word in items[i].split(" "):

                        # mask word flight number 
                        if mask and i == 2 and ((intent_goal == 'book' and final_state == 'book') or (intent_goal == 'change' and final_state == 'change')):
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
                            if '1000' in word and ('ight' in words[-4:] or 'umber' in words[-4:]):
                                # word = '<mask_flight>'
                                word = '<fl_1000>'
                                count += 1
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

                        # if i < 3: # tokenize intent, action, dialogue
                        if i == 0 or i==2 : # tokenize intent, action, dialogue
                            try:
                                words.append(self.dictionary.word2idx[word])
                            except KeyError:
                                words.append(self.dictionary.word2idx[UNK])
                        elif i == 3: # tokenize boundaries
                            words.append(int(word))
                    sent.append(words)
                # 
                # a, b, c, d = sent[0], sent[1], sent[2], sent[3]
                sents.append(sent)
                sents_len.append(len(sent[2]))
                self.size += 1
            self.data = sents
        print('Count : ', 100. * count/len(sents))
        return sents, sents_len, true_answer, action_state

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
                for word in line.split():
                    try:
                        words.append(self.dictionary.word2idx[word])
                    except KeyError:
                        words.append(self.dictionary.word2idx[UNK])
                        print('tokenize_kb : ', word)
                        raise
                for i in range(13):
                    words.append(empty_id)
                kb_sents.append(words)
                self.kb_size += 1
            self.kb_data = kb_sents
        return kb_sents

    def tokenize_kb_Info(self):
        Info =  ['<a1_IAD>', '<a2_ATL>', '<a1_MSP>', '<a2_IAD>', '<a2_MSP>', '<a1_ATL>', '<a1_HOU>', '<a2_HOU>', '<a2_LAS>', '<a1_LAS>', '<a2_ORD>', '<a1_DEN>', '<a2_DEN>', '<a1_ORD>', '<a1_OAK>', '<a2_JFK>', '<a1_PHL>', '<a2_PHL>', '<a1_JFK>', '<a2_OAK>', '<a1_PHX>', '<a2_AUS>', '<a1_AUS>', '<a2_PHX>', '<a1_BOS>', '<a2_DTW>', '<a1_SEA>', '<a2_BOS>', '<a1_DTW>', '<a2_SEA>', '<a1_IAH>', '<a2_IAH>', '<a1_LGA>', '<a2_LGA>', '<a2_CLT>', '<a1_CLT>', '<a1_SFO>', '<a2_SFO>', '<a2_DCA>', '<a1_DCA>', '<a2_DFW>', '<a1_DFW>', '<a2_MCO>', '<a1_MCO>', '<a2_EWR>', '<a1_EWR>', '<a2_LAX>', '<a1_LAX>', \
                '<m1_Sept>', '<m2_Sept>', '<m1_Apr>', '<m2_Apr>', '<m1_Feb>', '<m2_Feb>', '<m1_Jan>', '<m1_Aug>', '<m2_Aug>', '<m1_June>', '<m2_June>', '<m2_Jan>', '<m1_Nov>', '<m2_Nov>', '<m1_Oct>', '<m2_Oct>', '<m1_May>', '<m2_May>', '<m1_July>', '<m2_July>', '<m2_Mar>', '<m1_Mar>', '<m1_Dec>', '<m2_Dec>', \
                '<d1_11>', '<d2_13>', '<d2_14>', '<d1_12>', '<d1_10>', '<d2_12>', '<d1_13>', '<d1_3>', '<d2_6>', '<d2_5>', '<d1_4>', '<d2_4>', '<d1_5>', '<d1_2>', '<d1_1>', '<d2_3>', '<d1_31>', '<d1_15>', '<d2_16>', '<d2_17>', '<d1_14>', '<d2_18>', '<d1_21>', '<d2_23>', '<d1_22>', '<d2_24>', '<d1_23>', '<d2_25>', '<d2_22>', '<d1_16>', '<d1_17>', '<d2_20>', '<d2_19>', '<d1_18>', '<d1_25>', '<d2_26>', '<d1_24>', '<d2_28>', '<d2_27>', '<d1_26>', '<d1_20>', '<d2_21>', '<d2_7>', '<d1_6>', '<d1_7>', '<d2_8>', '<d1_19>', '<d1_9>', '<d2_10>', '<d2_11>', '<d1_8>', '<d1_30>', '<d2_2>', '<d2_1>', '<d2_31>', '<d1_29>', '<d2_9>', '<d1_27>', '<d2_29>', '<d2_30>', '<d1_28>', '<d2_15>', \
                '<tn1_10>', '<tn2_21>', '<tn1_21>', '<tn2_0>', '<tn2_6>', '<tn2_2>', '<tn1_13>', '<tn2_20>', '<tn1_1>', '<tn2_15>', '<tn1_8>', '<tn1_6>', '<tn2_5>', '<tn1_23>', '<tn2_12>', '<tn2_14>', '<tn1_14>', '<tn1_0>', '<tn1_7>', '<tn2_18>', '<tn1_3>', '<tn2_17>', '<tn2_8>', '<tn1_17>', '<tn1_4>', '<tn1_5>', '<tn1_12>', '<tn2_16>', '<tn2_7>', '<tn1_11>', '<tn2_1>', '<tn1_2>', '<tn2_23>', '<tn1_9>', '<tn2_11>', '<tn2_4>', '<tn1_20>', '<tn1_16>', '<tn2_22>', '<tn2_13>', '<tn2_3>', '<tn2_10>', '<tn1_18>', '<tn2_9>', '<tn1_19>', '<tn2_19>', '<tn1_22>', '<tn1_15>', \
                '<cl_business>', '<cl_economy>', \
                '<pr_800>', '<pr_200>', '<pr_100>', '<pr_500>', '<pr_600>', '<pr_400>', '<pr_300>', '<pr_1100>', '<pr_900>', '<pr_700>', '<pr_1000>', '<pr_1300>', '<pr_1200>', '<pr_1600>', '<pr_1400>', '<pr_1500>', '<pr_1700>', '<pr_1800>', '<pr_1900>', '<pr_2000>', '<pr_2100>', '<pr_2200>', \
                '<cn_1>', '<cn_0>', '<cn_2>', \
                '<al_AA>', '<al_UA>', '<al_Delta>', '<al_Southwest>', '<al_Frontier>', '<al_Spirit>', '<al_JetBlue>', '<al_Hawaiian>']
        words = []
        for i in Info:
            try:
                words.append(self.dictionary.word2idx[i])
            except KeyError:
                print(i, ' Not in dict !')
        return words

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

def read_state_track(stk_path):
    with open(stk_path, 'r') as f:
        all_state_list = []
        all_SQL_YN = []
        for line in f:
            line = line.replace(',', '').replace('\'', '')
            items = line.split("|")
            state_list = []

            SQL_YN = 1
            if items[2].split()[0] == 'Y_SQL':
                SQL_YN = 1
            elif items[2].split()[0] == 'N_SQL':
                SQL_YN = 0
            else:
                print('SQL ERROR !')
                raise
            all_SQL_YN.append(SQL_YN)

            for word in items[0].split():
                if word == '?':
                    print('ERROR : read_state_track meet "?" ')
                    raise
                else:
                    state_list.append(int(word))
            all_state_list.append(state_list)
    return all_state_list, all_SQL_YN

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

def torch_sequence_mask(lengths, maxlen):
    """Suppose lengths is a 1d vector and maclen is a scalar"""
    mask = np.zeros( (len(lengths), maxlen), dtype=bool)
    for i in range(len(lengths)):
        mask[i][0:lengths[i]] = True
    return mask

def process_entry_common(intent, action, dialogue, boundaries, kb, vocab_table, t1_id, t2_id, sql_data, table_data, state_tracking_data, SQL_YN):
    """A common procedure to process each entry of the dialogue data."""

    def do_process_boundary(start_points, end_points, input_length, t1_id, t2_id, all_tokenlized_diag):
        """function that contains the majority of the logic to proess boundary."""

        masks_start = torch_sequence_mask(start_points, input_length)
        masks_end = torch_sequence_mask(end_points, input_length)
        xor_masks = np.logical_xor(masks_start, masks_end)
        mask1 = np.any(xor_masks, axis=0)
        mask2 = np.logical_not(mask1)
        all_turn1 = np.equal(all_tokenlized_diag, t1_id) # compute element-wise equal
        all_turn2 = np.equal(all_tokenlized_diag, t2_id) # compute element-wise equal
        turn_point = np.logical_or(all_turn1, all_turn2)
        turn_point = turn_point.astype(np.float32) # logical or and convert type to float
        # print('start_points : ', start_points)
        # print('end_points : ', end_points)
        # print('masks_start : ', masks_start)
        # print('masks_end : ', masks_end)
        # print('xor_masks : ', xor_masks)
        return mask1, mask2, turn_point

    def process_boundary(boundaries, input_length, t1_id, t2_id, all_dialogue):
        """process the boundaries of the dialogue."""
        # points = boundaries.split(" ")
        # points_val = [int(i) for i in points]
        points_val = boundaries
        siz = len(points_val) // 2
        start_points, end_points = points_val[0:siz], points_val[siz:]
        return do_process_boundary(start_points, end_points, input_length, t1_id, t2_id, all_dialogue)

    def process_dialogue(dialogue, size_dialogue, mask1, mask2, turn_point, state_tracking_data, gate_label, eval_step):
        new_dialogue_size = size_dialogue - 1
        source = dialogue[0:-1]
        target = dialogue[1:]
        mask1 = mask1[0:-1]
        mask2 = mask2[0:-1]
        turn_point = turn_point[0:-1]
        state_tracking_data = state_tracking_data[0:-1]
        gate_label = gate_label[0:-1]
        eval_step = eval_step[0:-1]
        return source, target, new_dialogue_size, mask1, mask2, turn_point, state_tracking_data, gate_label, eval_step

    def process_sql_table(sql_data, table_data):

        sql = sql_data
        ans_seq =(len(sql['sql']['conds']), tuple(x[0] for x in sql['sql']['conds']), tuple(x[1] for x in sql['sql']['conds']), tuple(x[2] for x in sql['sql']['conds'])) # (x[0]) column list , (x[1]) operator list
        col_seq = table_data[sql['table_id']]['header_tok']
        col_num = len(table_data[sql['table_id']]['header_tok'])
        col_name_len = table_data[sql['table_id']]['header_len']
        truth_seq = sql['query']

        return col_seq, col_num, col_name_len, ans_seq, truth_seq

    def process_state_tracking(mask1, mask2, turn_point, state_tracking_data, size_dialogue, SQL_YN):
        # print('mask1 : ', mask1)
        # print('mask2 : ', mask2)
        # print('turn_point : ', turn_point.shape)
        # print('state_tracking_data : ', state_tracking_data)
        # print('size_dialogue : ', size_dialogue)
        state_tracking_label = np.zeros_like(turn_point)
        state_tracking_label[state_tracking_data] = 1
        gate_label = np.zeros_like(turn_point)
        if SQL_YN != 0:
            gate_label[state_tracking_data[0]:] = 1
        # for i in range(gate_label.shape[0]):
        #     if i > state_tracking_data[0] and mask2[i] == True:
        #         gate_label[i] = 1
        # print('state_tracking_label : ', state_tracking_label, state_tracking_label.shape)
        # print('gate_label : ', gate_label)
        eval_step = np.ones_like(turn_point)
        eval_step[state_tracking_data[0]+1:] = 0

        return state_tracking_label, gate_label, state_tracking_data, eval_step

    size_intent = len(intent)
    size_dialogue = len(dialogue)
    predicted_action = action + [2]
    action = [1] + action
    size_action = len(action)
    has_reservation = kb[0]
    kb = kb[1:]
    mask1, mask2, turn_point = process_boundary(boundaries, size_dialogue, t1_id, t2_id, dialogue)
    state_tracking_data, gate_label, state_tracking_list, eval_step = process_state_tracking(mask1, mask2, turn_point, state_tracking_data, size_dialogue, SQL_YN)
    source_diag, target_diag, size_dialogue, mask1, mask2, turn_point, state_tracking_data, gate_label, eval_step = process_dialogue(dialogue, size_dialogue, mask1, mask2, turn_point, state_tracking_data, gate_label, eval_step)

    col_seq, col_num, col_name_len, ans_seq, truth_seq = process_sql_table(sql_data, table_data)

    return intent, size_intent, source_diag, target_diag, size_dialogue, action, predicted_action, size_action, kb, has_reservation, \
    mask1, mask2, turn_point, col_seq, col_num, col_name_len, ans_seq, truth_seq, state_tracking_data, gate_label, state_tracking_list, eval_step

def process_entry_supervised(inputs, vocab_table, t1_id, t2_id, table_data):
    """Pre-process procedure for the supervised iterator."""
    intent, action, dialogue, boundaries, kb, sql_data, state_tracking_data, SQL_YN, Info, kb_true_answer, filter_index = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10]
    res = process_entry_common(intent, action, dialogue, boundaries, kb, vocab_table, t1_id, t2_id, sql_data, table_data, state_tracking_data, SQL_YN)
    intent, size_intent, source_diag, target_diag, size_dialogue, action, predicted_action, size_action, kb, has_reservation, \
        mask1, mask2, turn_point, col_seq, col_num, col_name_len, ans_seq, truth_seq, state_tracking_data, gate_label, state_tracking_list, eval_step = res

    return intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation,\
        predicted_action, 0.0, 0.0, mask1, mask2, turn_point, col_seq, col_num, col_name_len, ans_seq, truth_seq, state_tracking_data, gate_label, state_tracking_list, SQL_YN, Info, kb_true_answer, eval_step, filter_index

def process_entry_inferenced(inputs, vocab_table, t1_id, t2_id, table_data):
    """Pre-process procedure for the supervised iterator."""
    intent, dialogue, kb, tar_dialogue, kb_true_answer = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

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

    return intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer

def process_entry_self_play_eval(inputs, vocab_table, t1_id, t2_id, table_data):
    """Pre-process procedure for the supervised iterator."""
    intent, action, kb, kb_true_answer, SQL_YN, sql_data, filter_index  = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]

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

    return intent, size_intent, action, kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, kb_true_answer, SQL_YN, filter_index

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

def collate_fn(data):
    global eod_id
    global smallkb_n
    global empty_id
    data.sort(key=lambda x: len(x[2]), reverse=True)
    intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation, \
        predicted_action, reward_diag, reward_action, mask1, mask2, turn_point, col_seq, col_num, col_name_len, \
        ans_seq, truth_seq, state_tracking_data, gate_label, state_tracking_list, SQL_YN, Info, kb_true_answer_1, eval_step, filter_index = zip(*data)

    # pad
    # print('col_seq2 : ', len(col_seq), col_seq)
    # print('col_num2 : ', len(col_num), col_num)
    # print('col_name_len2 : ', len(col_name_len), col_name_len)

    intent_pad = pad_sequence(list_to_tensor(intent), batch_first=True, padding_value=eod_id)
    size_intent = torch.tensor(size_intent, dtype=torch.int64)
    # size_intent_pad = pad_sequence(list_to_tensor(size_intent), batch_first=True, padding_value=0) # seems needless
    source_diag_pad = pad_sequence(list_to_tensor(source_diag), batch_first=True, padding_value=eod_id)
    target_diag_pad = pad_sequence(list_to_tensor(target_diag), batch_first=True, padding_value=eod_id)
    size_dialogue = torch.tensor(size_dialogue, dtype=torch.int64)
    # size_dialogue_pad = pad_sequence(torch.tensor(size_dialogue), batch_first=True, padding_value=0) # seems needless
    # action_pad = pad_sequence(torch.tensor(action), batch_first=True, padding_value=eod_id)
    action_pad = torch.tensor(action, dtype=torch.int64)
    size_action = torch.tensor(size_action, dtype=torch.int64)
    # size_action_pad = pad_sequence(torch.tensor(size_action), batch_first=True, padding_value=0)  # seems needless
    
    kb = torch.tensor(kb, dtype=torch.int64).view(-1, 31, 13)
    kb_true_answer = torch.tensor(kb_true_answer_1, dtype=torch.int64)
    kb_true_answer_onehot = torch.zeros(kb_true_answer.size(0), 31).scatter_(1, kb_true_answer.view(kb_true_answer.size(0), 1).data, 1)
    # kb_true_answer_onehot = kb_true_answer_onehot[:, :-1]
    # print('1. kb_true_answer : ', kb_true_answer)
    # print('1. kb_true_answer_onehot : ', kb_true_answer_onehot)
    ######################## small kb ###################################

    filter_index_tensor = pad_sequence(list_to_tensor(filter_index), batch_first=True, padding_value=0) # (b, 30)
    kb_true_answer_2 = torch.tensor(kb_true_answer_1, dtype=torch.int64)
    filter_index_tensor_expand = filter_index_tensor.unsqueeze(2).expand(-1, -1, 13)
    filter_kb = filter_index_tensor_expand * kb

    ######################## test different kb ###################################
    # print('kb : ', kb.size())
    # print('kb_true_answer_onehot : ', kb_true_answer_onehot.size())
    # print('kb_true_answer_onehot : ', kb_true_answer_onehot.size())
    # print('2. kb_true_answer : ', kb_true_answer)
    # print('2. kb : ', kb)
    # print('2. kb_true_answer_onehot : ', kb_true_answer_onehot)
    # raise
    # kb_shuffle = list(range(kb.size(1)))
    # random.shuffle(kb_shuffle)
    # kb = kb[:, kb_shuffle]
    # kb = torch.zeros_like(kb)
    # kb_shuffle_b = list(range(kb.size(0)))
    # kb_pad = kb
    # for i in range(10):
    #     random.shuffle(kb_shuffle_b)
    #     kb_pad = torch.cat((kb_pad, kb[kb_shuffle_b]), dim=1)
    # kb= kb_pad

    has_reservation = torch.tensor(has_reservation, dtype=torch.int64)
    predicted_action = torch.tensor(predicted_action, dtype=torch.int64)
    # predicted_action_pad = pad_sequence(list_to_tensor(predicted_action), batch_first=True, padding_value=eod_id) ## double check
    reward_diag = torch.tensor(reward_diag, dtype=torch.int64)
    # reward_diag_pad = pad_sequence(reward_diag, batch_first=True, padding_value=0)
    reward_action = torch.tensor(reward_action, dtype=torch.int64)
    # reward_action_pad = pad_sequence(reward_action, batch_first=True, padding_value=0)
    mask1_pad = pad_sequence(list_to_tensor(mask1), batch_first=True, padding_value=0)
    mask2_pad = pad_sequence(list_to_tensor(mask2), batch_first=True, padding_value=0)
    turn_point_pad = pad_sequence(list_to_tensor(turn_point), batch_first=True, padding_value=0)
    state_tracking_pad = pad_sequence(list_to_tensor(state_tracking_data), batch_first=True, padding_value=0)
    gate_label = pad_sequence(list_to_tensor(gate_label), batch_first=True, padding_value=0)
    eval_step = pad_sequence(list_to_tensor(eval_step), batch_first=True, padding_value=0)
    SQL_YN_tensor = torch.tensor(SQL_YN, dtype=torch.int64)
    Info = torch.tensor(Info, dtype=torch.int64)

    col_seq = torch.tensor(col_seq) # (b, 12, 1)
    col_num = torch.tensor(col_num, dtype=torch.int64)

    return intent_pad, size_intent, source_diag_pad, target_diag_pad, size_dialogue, action_pad, \
        size_action, filter_kb, has_reservation, predicted_action, reward_diag, reward_action, mask1_pad, mask2_pad, turn_point_pad, \
        col_seq, col_num, ans_seq, truth_seq, state_tracking_pad, gate_label, state_tracking_list, SQL_YN_tensor, Info, kb_true_answer_onehot, eval_step, kb_true_answer_2

def collate_fn_inference(data):
    global eod_id
    global smallkb_n
    data.sort(key=lambda x: len(x[2]), reverse=True)
    intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer = zip(*data)

    intent_pad = pad_sequence(list_to_tensor(intent), batch_first=True, padding_value=eod_id)
    size_intent = torch.tensor(size_intent, dtype=torch.int64)
    source_diag_pad = pad_sequence(list_to_tensor(source_diag), batch_first=True, padding_value=eod_id)
    target_diag_pad = pad_sequence(list_to_tensor(target_diag), batch_first=True, padding_value=eod_id)
    size_dialogue = torch.tensor(size_dialogue, dtype=torch.int64)
    kb = torch.tensor(kb, dtype=torch.int64).view(-1, 31, 13)
    has_reservation = torch.tensor(has_reservation, dtype=torch.int64)
    col_seq = torch.tensor(col_seq) # (b, 12, 1)
    col_num = torch.tensor(col_num, dtype=torch.int64)
    kb_true_answer = torch.tensor(kb_true_answer, dtype=torch.int64)

    return intent_pad, size_intent, source_diag_pad, target_diag_pad, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer

def collate_fn_self_play_eval(data):
    global eod_id
    global smallkb_n
    intent, size_intent, action, kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, kb_true_answer, SQL_YN, filter_index = zip(*data)

    intent_pad = pad_sequence(list_to_tensor(intent), batch_first=True, padding_value=eod_id)
    size_intent = torch.tensor(size_intent, dtype=torch.int64)
    action_pad = torch.tensor(action, dtype=torch.int64)
    
    kb = torch.tensor(kb, dtype=torch.int64).view(-1, 31, 13)
    kb_true_answer = torch.tensor(kb_true_answer, dtype=torch.int64)
    kb_true_answer_onehot = torch.zeros(kb_true_answer.size(0), 31).scatter_(1, kb_true_answer.view(kb_true_answer.size(0), 1).data, 1)

    filter_index_tensor = pad_sequence(list_to_tensor(filter_index), batch_first=True, padding_value=0) # (b, 30)
    filter_index_tensor_expand = filter_index_tensor.unsqueeze(2).expand(-1, -1, 13)
    filter_kb = filter_index_tensor_expand * kb

    has_reservation = torch.tensor(has_reservation, dtype=torch.int64)
    SQL_YN_tensor = torch.tensor(SQL_YN, dtype=torch.int64)
    col_seq = torch.tensor(col_seq) # (b, 12, 1)
    col_num = torch.tensor(col_num, dtype=torch.int64)

    # return intent_pad, size_intent, action_pad, filter_kb, has_reservation, col_seq, col_num, ans_seq, truth_seq, SQL_YN_tensor, kb_true_answer_onehot, kb_true_answer 
    return intent_pad, size_intent, action_pad, filter_kb, has_reservation, col_seq, truth_seq, SQL_YN_tensor, kb_true_answer

def loader(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30):
    global eod_id
    global smallkb_n
    global empty_id
    smallkb_n = small_n
    print('Kb_n : ', smallkb_n)
    # test
    vocab_file = 'data/synthesized/tokenized/vocab.txt'
    if dev:
        data_file = 'data/synthesized/tokenized/dev.eval.data'
        kb_file = 'data/synthesized/tokenized/dev.eval.kb'
        table_path = 'data/synthesized/SQL/dev/train_tok.tables.jsonl'
        sql_path = 'data/synthesized/SQL/dev/dev_tok.jsonl'
        stk_path = 'data/synthesized/SQL/dev/State_Tracking.txt'
        filter_kb = 'data/synthesized/SQL/dev/filtered_kb'
    else:
        data_file = 'data/synthesized/tokenized/train.data'
        kb_file = 'data/synthesized/tokenized/train.kb'
        table_path = 'data/synthesized/SQL/train_tok.tables.jsonl'
        sql_path = 'data/synthesized/SQL/train_tok.jsonl'
        stk_path = 'data/synthesized/SQL/State_Tracking.txt'
        filter_kb = 'data/synthesized/SQL/filtered_kb'

    # vocab table & tokenize
    corpus = Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    print('Use mask : ', mask)
    sents, sents_len, kb_true_answer, action_state = corpus.tokenize(data_file, mask)
    kb_sents = corpus.tokenize_kb(kb_file)
    Info = corpus.tokenize_kb_Info()
    sql_data, table_data = corpus.tokenize_column(sql_path, table_path)
    state_tracking_data, SQL_YN = read_state_track(stk_path)
    filter_index = read_fileter_kb(filter_kb)
    # print('sents : ', len(sents), ' kb_sents : ', len(kb_sents))

    if n_sample != -1:
        sents = sents[:n_sample]
        sents_len = sents_len[:n_sample]
        kb_true_answer = kb_true_answer[:n_sample]
        action_state = action_state[:n_sample]
        kb_sents = kb_sents[:n_sample]
        sql_data = sql_data[:n_sample]
        state_tracking_data = state_tracking_data[:n_sample]
        SQL_YN = SQL_YN[:n_sample]
        filter_index = filter_index[:n_sample]

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(sql_data[i])
        combined_dataset[i].append(state_tracking_data[i])
        combined_dataset[i].append(SQL_YN[i])
        combined_dataset[i].append(Info)
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(filter_index[i])

    if only_f:
        combined_dataset_new = []
        sents_len_new = []
        for i in range(len(sents)):
            if action_state[i] != -1 and SQL_YN[i] == 1:
                combined_dataset_new.append(combined_dataset[i])
                sents_len_new.append(sents_len[i])
        print('combined_dataset new : ', len(combined_dataset_new), 100. * len(combined_dataset_new) / len(combined_dataset))
        combined_dataset = combined_dataset_new
        sents_len = sents_len_new
    # plot_hist(sents_len)

    # print('before sents_len : ', sents_len)
    sort_indices = sorted(range(len(sents_len)), key=lambda k: sents_len[k], reverse=True)
    sents_len_sort = []
    for i in sort_indices:
       sents_len_sort.append(sents_len[i]) 
    print('Max len in Dataset : ', sents_len_sort[0])

    if max_len is not None:
        print('Filter out > max_len : ', max_len)
    combined_dataset_sort = []
    for i in sort_indices:
        if max_len is not None:
            if sents_len[i] < max_len:
                combined_dataset_sort.append(combined_dataset[i]) 

    print('Before Filtering combined_dataset : ', len(combined_dataset))
    print('After  Filtering combined_dataset : ', len(combined_dataset_sort))
    # print('combined_dataset[0] : ', len(combined_dataset[0]))

    eod_id= corpus.dictionary.word2idx['<eod>']
    t1_id = corpus.dictionary.word2idx['<t1>']
    t2_id = corpus.dictionary.word2idx['<t2>']
    unk = corpus.dictionary.word2idx['<unk>']
    empty_id = corpus.dictionary.word2idx['<empty_id>']
    combined_dataset = map(partial(process_entry_supervised, vocab_table=corpus, t1_id=t1_id, t2_id=t2_id, table_data=table_data), combined_dataset_sort)

    # print('len combined_dataset : ', len(combined_dataset))
    data = AirData(combined_dataset)
    if dev :
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=need_shuffle, collate_fn=collate_fn, drop_last=False, num_workers=8, pin_memory=True)
    
    # batch_x = iter(data_loader).next()
    # print('LEN : ', batch_x[4])

    return data_loader, corpus
    
def Inference_loader(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30):
    global eod_id
    global smallkb_n
    smallkb_n = small_n
    print('Kb_n : ', smallkb_n)
    # test
    vocab_file = 'data/synthesized/tokenized/vocab.txt'
    if dev:
        src_data_file = 'data/synthesized/tokenized/dev.infer.src.data'
        tar_data_file = 'data/synthesized/tokenized/dev.infer.tar.data'
        kb_file = 'data/synthesized/tokenized/dev.infer.kb'
        table_path = 'data/synthesized/SQL/dev/train_tok.tables.jsonl'
    else:
        src_data_file = 'data/synthesized/tokenized/dev.infer.src.data'
        tar_data_file = 'data/synthesized/tokenized/dev.infer.tar.data'
        kb_file = 'data/synthesized/tokenized/dev.infer.kb'
        table_path = 'data/synthesized/SQL/dev/train_tok.tables.jsonl'

    # vocab table & tokenize
    corpus = Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    print('Use mask : ', mask)
    sents, sents_len, tar_sents, kb_true_answer = corpus.infer_tokenize(src_data_file, tar_data_file, mask)
    print('Reading kb')
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(None, table_path)
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

def SelfPlayEval_loader(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30):
    global eod_id
    global smallkb_n
    smallkb_n = small_n
    print('Kb_n : ', smallkb_n)
    # test
    vocab_file = 'data/synthesized/tokenized/vocab.txt'
    src_data_file = 'data/synthesized/tokenized/dev.selfplay.eval.data'
    kb_file = 'data/synthesized/tokenized/dev.selfplay.eval.kb'
    table_path = 'data/synthesized/SQL/dev_self_play_eval/train_tok.tables.jsonl'
    sql_path = 'data/synthesized/SQL/dev_self_play_eval/self_play_eval_tok.jsonl'
    filter_kb = 'data/synthesized/SQL/dev/filtered_kb'
    print('SelfPlayEval_loader Loading data : ', src_data_file)
    print('SelfPlayEval_loader Loading kb : ', kb_file)

    # vocab table & tokenize
    corpus = Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    sents, kb_true_answer, action_state, SQL_YN = corpus.self_play_eval_tokenize(src_data_file)
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(sql_path, table_path)
    filter_index = read_fileter_kb(filter_kb)

    if n_sample != -1:
        sents = sents[:n_sample]
        kb_sents = kb_sents[:n_sample]
        SQL_YN = SQL_YN[:n_sample]
        kb_true_answer = kb_true_answer[:n_sample]
        sql_data = sql_data[:n_sample]
        filter_index = filter_index[:n_sample]

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(SQL_YN[i])
        combined_dataset[i].append(sql_data[i])
        combined_dataset[i].append(filter_index[i])

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

def plot_hist(sent):
    # print(sent)
    plt.hist(x=sent, bins='auto', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.show()

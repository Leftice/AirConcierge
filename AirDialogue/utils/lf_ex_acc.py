import pickle
import os
import argparse
from tqdm import tqdm 
import numpy as np
import time

# python3 different_augment.py --dev --aug_n 10 --fix '1 1 1 1 1 1 0 0 0 0 0 0'
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--aug_n', default=10, type=int, help='augment number')
parser.add_argument('--sample', default=200, type=int, help='augment number')
parser.add_argument('--fix', default='0 0 0 0 0 0 0 0 0 0 0 0', type=str, help='augment number')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
parser.add_argument('--toy', action='store_true', help='Use toy')
global args
args = parser.parse_args()

if args.syn:
    data_path = './results/synthesized/'
    data_path2 = './data/synthesized/'
elif args.air:
    data_path = './results/airdialogue/'
    data_path2 = './data/airdialogue/'
else:
    print('Pleae use --syn or --air !')
    raise

if args.dev:
    if args.toy:
        data_file = data_path2 + 'tokenized/sub_air/toy_dev.eval.data'
        kb_file =  data_path2 + 'tokenized/sub_air/toy_dev.eval.kb'
    else:
        data_file = data_path2 + 'tokenized/dev/dev.eval.data'
        kb_file =  data_path2 + 'tokenized/dev/dev.eval.kb'
    p_query_file = data_path + 'SQL/dev_sql/dev_simple_predict'
    g_query_file = data_path + 'SQL/dev_sql/dev_simple_ground'
else:
    print('Please use --dev !')
    raise

def tokenize_dialogue(path):
    sents = []
    sents_len = [] 
    with open(path, 'r') as f:
        for line in f:
            items = line.split("|")
            sent = []
            for i in range(4):
                words = []
                for word in items[i].split(" "):
                    if i < 3: # tokenize intent, action, dialogue
                        words.append(word)
                    else: # tokenize boundaries
                        words.append(int(word))
                sent.append(words)
            # a, b, c, d = sent[0], sent[1], sent[2], sent[3]
            sents.append(sent)
            sents_len.append(len(sent[2]))
    return sents, sents_len

def tokenize_kb(path):
    # <res_no_res> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_21> <cl_business> <pr_800> <cn_1> <al_AA> <fl_1000> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_21> <tn2_0> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1001> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_21> <tn2_6> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1002> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_21> <tn2_2> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1003> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_13> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1004> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_1> <tn2_15> <cl_economy> <pr_100> <cn_0> <al_Frontier> <fl_1005> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_12> <tn1_8> <tn2_21> <cl_economy> <pr_200> <cn_1> <al_Delta> <fl_1006> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_6> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1007> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_14> <tn1_23> <tn2_12> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1008> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_21> <tn2_14> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1009> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_12> <cl_business> <pr_500> <cn_1> <al_Southwest> <fl_1010> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1011> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_13> <d2_12> <tn1_0> <tn2_21> <cl_economy> <pr_200> <cn_0> <al_UA> <fl_1012> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_7> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1013> <a1_ATL> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_7> <tn2_0> <cl_economy> <pr_200> <cn_1> <al_AA> <fl_1014> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_6> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1015> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_10> <d2_13> <tn1_23> <tn2_18> <cl_economy> <pr_200> <cn_1> <al_Hawaiian> <fl_1016> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_3> <tn2_17> <cl_economy> <pr_200> <cn_1> <al_Spirit> <fl_1017> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_10> <tn2_8> <cl_economy> <pr_200> <cn_1> <al_JetBlue> <fl_1018> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_17> <tn2_14> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1019> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_20> <cl_economy> <pr_100> <cn_1> <al_Delta> <fl_1020> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_5> <tn2_15> <cl_economy> <pr_200> <cn_1> <al_Southwest> <fl_1021> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_12> <tn1_12> <tn2_5> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1022> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_14> <tn2_16> <cl_economy> <pr_100> <cn_1> <al_Southwest> <fl_1023> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_4> <tn2_7> <cl_economy> <pr_100> <cn_1> <al_Spirit> <fl_1024> <a1_MSP> <a2_ATL> <m1_Sept> <m2_Sept> <d1_12> <d2_13> <tn1_11> <tn2_16> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1025> <a1_IAD> <a2_MSP> <m1_Sept> <m2_Sept> <d1_12> <d2_14> <tn1_8> <tn2_1> <cl_economy> <pr_100> <cn_1> <al_Hawaiian> <fl_1026> <a1_MSP> <a2_IAD> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_5> <cl_economy> <pr_200> <cn_1> <al_UA> <fl_1027> <a1_IAD> <a2_ATL> <m1_Sept> <m2_Sept> <d1_11> <d2_14> <tn1_17> <tn2_23> <cl_economy> <pr_100> <cn_1> <al_UA> <fl_1028> <a1_ATL> <a2_MSP> <m1_Sept> <m2_Sept> <d1_11> <d2_13> <tn1_2> <tn2_20> <cl_economy> <pr_200> <cn_1> <al_Frontier> <fl_1029>
    kb_sents = []
    reservation = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            kb = []
            entry=[]
            i=0
            reservation.append(line.split()[0])
            for word in line.split()[1:]: # without reservation
                entry.append(word)
                i += 1
                if i % 13 == 0:
                    kb.append(entry)
                    entry = []    
            kb_sents.append(kb) 
    return kb_sents, reservation

def tokenize_query(path):
    query = []
    gate = []
    with open(path, 'r') as f:
        for line in f:
            items = line.split("|")
            gate.append(int(items[0]))
            words = []
            for word in items[1].split():
                words.append(int(word))
            query.append(words)
    return query, gate

def MyDBMS(query, kb, condition_num):
    
    # <a1_MCO> <a2_LGA> <m1_Feb> <m2_Feb> <d1_9> <d2_10> <tn1_2> <tn2_7> <cl_business> <pr_400> <cn_1> <al_Southwest> <fl_1000>
    airport_list = ['DEN', 'LAX', 'MSP', 'DFW', 'SEA', 'ATL', 'IAH', 'DTW', 'ORD', 'IAD', 'CLT', 'EWR', 'LGA', 'JFK', 'HOU', 'SFO', 'AUS', 'OAK', 'LAS', 'PHL', 'BOS', 'MCO', 'DCA', 'PHX']
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    time_list = ['morning', 'afternoon', 'evening']
    air_class_list = ['economy', 'business']
    max_price_list = ['200', '500', '1000', '5000']
    airline_preference_list = ['normal-cost']

    filtered_flight = []
    filtered_kb = []
    filtered_index = [0 for _ in range(30)]
    total = 0
    for entry in range(len(kb)):
        correct = 1
        for c in range(len(kb[entry])-1):
            if query[c] == -1:
                continue 
            if c >= condition_num:
                break
            token = kb[entry][c].split('_', 1)[1].split('>', 1)[0]
            if c == 0:
                airport_index = airport_list.index(token)
                if airport_index != query[c]:
                    correct = 0
                    break
            elif c == 1:
                airport_index = airport_list.index(token)
                if airport_index != query[c]:
                    correct = 0
                    break
            elif c == 2:
                month_index = month_list.index(token)
                if month_index != query[c]:
                    correct = 0
                    break
            elif c == 3:
                month_index = month_list.index(token)
                if month_index != query[c]:
                    correct = 0
                    break
            elif c == 4:
                if int(token)-1 != query[c]:
                    correct = 0
                    break
            elif c == 5:
                if int(token)-1 != query[c]:
                    correct = 0
                    break
            elif c == 6:
                d_time = time_list[query[c]]
                if d_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    correct = 0
                    break
                if d_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                    correct = 0
                    break
                if d_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                    correct = 0
                    break
            elif c == 7:
                r_time = time_list[query[c]]
                if r_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    correct = 0
                    break
                if r_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                    correct = 0
                    break
                if r_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                    correct = 0
                    break
            elif c == 8:
                class_index = air_class_list.index(token)
                if class_index != query[c]:
                    correct = 0
                    break
            elif c == 9:
                if int(max_price_list[query[c]]) < int(token):
                    correct = 0
                    break
            elif c == 10:
                if query[c] < int(token):
                    correct = 0
                    break
            elif c == 11:
                if query[c] == 1 and token not in ['UA', 'AA', 'Delta', 'Hawaiian']:
                    correct = 0
                    break
        if correct == 1:
            total += 1
            filtered_kb.append(kb[entry])
            filtered_flight.append(kb[entry][12].split('_', 1)[1].split('>', 1)[0])
            filtered_index[entry] = 1

    return filtered_index, total

def ACC_ex(g_gate, p_gate, p_filtered_flight, g_filtered_flight, ACC_ex_correct, ACC_ex_total):
    # Just compare two query even gate=0 & query=None
    # Only consider ground truth gate = 1
    if g_gate == 1:
        if p_filtered_flight == g_filtered_flight:
            ACC_ex_correct += 1
        ACC_ex_total += 1
    # Consider g_gate=1 & g_gate=0 but p_gate=1
    
    return ACC_ex_correct, ACC_ex_total

def ACC_lf(g_gate, p_gate, g_query, p_query, ACC_lf_correct, ACC_lf_total):
    
    # Just compare two query even gate=0 & query=None
    for i in range(1,13):
        if g_query[:i] == p_query[:i]:
            ACC_lf_correct[0][i-1] += 1
    ACC_lf_total[0] += 1

    # Only consider ground truth gate = 1
    if g_gate == 1:
        if p_gate != 0:
            for i in range(1,13):
                if g_query[:i] == p_query[:i]:
                    ACC_lf_correct[1][i-1] += 1
        ACC_lf_total[1] += 1

    # Consider g_gate=1 & g_gate=0 but p_gate=1
    if g_gate == 1 and p_gate == 0:
        ACC_lf_total[2] += 1
    if g_gate == 1 and p_gate == 1:
        for i in range(1,13):
            if g_query[:i] == p_query[:i]:
               ACC_lf_correct[2][i-1] += 1 
        ACC_lf_total[2] += 1
    if g_gate == 0 and p_gate == 1:
        ACC_lf_total[2] += 1

    return ACC_lf_correct, ACC_lf_total

def ACC_lf(g_gate, p_gate, g_query, p_query, ACC_lf_correct, ACC_lf_total):
    
    # Just compare two query even gate=0 & query=None
    for i in range(1,13):
        if g_query[:i] == p_query[:i]:
            ACC_lf_correct[0][i-1] += 1
    ACC_lf_total[0] += 1

    # Only consider ground truth gate = 1
    if g_gate == 1:
        if p_gate != 0:
            for i in range(1,13):
                if g_query[:i] == p_query[:i]:
                    ACC_lf_correct[1][i-1] += 1
        ACC_lf_total[1] += 1

    # Consider g_gate=1 & g_gate=0 but p_gate=1
    if g_gate == 1 and p_gate == 0:
        ACC_lf_total[2] += 1
    if g_gate == 1 and p_gate == 1:
        for i in range(1,13):
            if g_query[:i] == p_query[:i]:
               ACC_lf_correct[2][i-1] += 1 
        ACC_lf_total[2] += 1
    if g_gate == 0 and p_gate == 1:
        ACC_lf_total[2] += 1

    return ACC_lf_correct, ACC_lf_total

def ACC_collf(g_gate, p_gate, g_query, p_query, ACC_collf_correct, ACC_collf_total, n):
    
    # Just compare two query even gate=0 & query=None
    if g_gate == 1:
        for i in range(n):
            if g_query[i] == p_query[i] and g_query[i] == -1: # NONE NONE
                ACC_collf_correct += 1
            elif g_query[i] != -1 and p_query[i] != -1: # 2 9, 1 1 ...
                ACC_collf_correct += 1
            ACC_collf_total += 1

    return ACC_collf_correct, ACC_collf_total

def ACC_val(g_gate, p_gate, g_query, p_query, ACC_val_correct, ACC_val_total):
    
    # Just compare two query even gate=0 & query=None
    for i in range(12):
        if g_query[i] == p_query[i]:
            ACC_val_correct[0][i] += 1
        ACC_val_total[0][i] += 1
    # Only consider ground truth gate = 1
    if g_gate == 1:
        for i in range(12):
            if g_query[i] == p_query[i]:
                ACC_val_correct[1][i] += 1
            ACC_val_total[1][i] += 1

    # Consider g_gate=1 & g_gate=0 but p_gate=1
    if g_gate == 1 and p_gate == 0:
        for i in range(12):
            if g_query[i] != -1:
                ACC_val_total[2][i] += 1
    if g_gate == 1 and p_gate == 1:
        # 4 combination : truth(y, y, y, n) predict(w, r, n, y)
        for i in range(12):
            # if g_query[i] != -1: # y
            #     if (p_query[i] != g_query[i]) and p_query[i] != -1: # w
            #         ACC_val_total[2][i] += 1
            #     if p_query[i] == g_query[i]: # r
            #         ACC_val_correct[2][i] += 1
            #         ACC_val_total[2][i] += 1
            #     if p_query[i] == -1: # n
            #         ACC_val_total[2][i] += 1
            # if g_query[i] == -1: # n
            #     if p_query[i] != -1: # y
            #         ACC_val_total[2][i] += 1
            if g_query[i] != -1 or  p_query[i] != -1:
                if p_query[i] == g_query[i]:
                    ACC_val_correct[2][i] += 1
                ACC_val_total[2][i] += 1

    if g_gate == 0 and p_gate == 1:
        for i in range(12):
            if p_query[i] != -1:
                ACC_val_total[2][i] += 1
    return ACC_val_correct, ACC_val_total

def simulate_DB(sents, kb, p_query, p_gate, g_query, g_gate, condition_num):
    global args
    
    max_kb = 0
    total = 0

    error, error_truth = 0, 0
    ACC_ex_correct = 0
    ACC_ex_total = 0
    ACC_val_correct = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ACC_val_total = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ACC_lf_correct = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ACC_lf_total = [0, 0, 0]
    ACC_collf_correct = 0
    ACC_collf_total = 0
    ACC_collf_correct2 = 0
    ACC_collf_total2 = 0
    ACC_gate = 0

    kb_len = [0 for _ in range(30)]

    for i in tqdm(range(len(p_query))):
        if g_gate[i] == p_gate[i]:
            ACC_gate += 1
        total += 1
        ACC_lf_correct, ACC_lf_total = ACC_lf(g_gate[i], p_gate[i], g_query[i], p_query[i], ACC_lf_correct, ACC_lf_total)
        ACC_collf_correct, ACC_collf_total = ACC_collf(g_gate[i], p_gate[i], g_query[i], p_query[i], ACC_collf_correct, ACC_collf_total, 6)
        ACC_collf_correct2, ACC_collf_total2 = ACC_collf(g_gate[i], p_gate[i], g_query[i], p_query[i], ACC_collf_correct2, ACC_collf_total2, 12)

        p_filtered_flight, p_total = MyDBMS(p_query[i], kb[i], condition_num)
        g_filtered_flight, g_total = MyDBMS(g_query[i], kb[i], condition_num)

        ACC_ex_correct, ACC_ex_total = ACC_ex(g_gate[i], p_gate[i], p_filtered_flight, g_filtered_flight, ACC_ex_correct, ACC_ex_total)
        ACC_val_correct, ACC_val_total = ACC_val(g_gate[i], p_gate[i], g_query[i], p_query[i] ,ACC_val_correct, ACC_val_total) 


    print('gate acc : ', ACC_gate , ' / ', total, ' -> ', 100.*ACC_gate /total)
    print('Acc ex : ', 100.*ACC_ex_correct/ACC_ex_total, ACC_ex_correct, ACC_ex_total)
    condiction_name = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', \
                 'price', 'num_connections', 'airline_preference']
    print('col acc 6 : ', ACC_collf_correct, ' / ', ACC_collf_total, ' -> ', 100.*ACC_collf_correct/ACC_collf_total)
    print('col acc 12: ', ACC_collf_correct2, ' / ', ACC_collf_total2, ' -> ', 100.*ACC_collf_correct2/ACC_collf_total2)
    for i in range(12):
        print()
        print('*'*100)
        print(condiction_name[i], ' sql_val0 : ', ACC_val_correct[0][i], ' / ', ACC_val_total[0][i], ' -> ', 100.*ACC_val_correct[0][i]/ACC_val_total[0][i])
        print(condiction_name[i], ' sql_val0 : ', ACC_val_correct[1][i], ' / ', ACC_val_total[1][i], ' -> ', 100.*ACC_val_correct[1][i]/ACC_val_total[1][i])
        print(condiction_name[i], ' sql_val0 : ', ACC_val_correct[2][i], ' / ', ACC_val_total[2][i], ' -> ', 100.*ACC_val_correct[2][i]/ACC_val_total[2][i])
        print('*'*100)
        print(condiction_name[i], ' lf0    : ', ACC_lf_correct[0][i], ' / ', ACC_lf_total[0], ' -> ', 100.*ACC_lf_correct[0][i]/ACC_lf_total[0])
        print(condiction_name[i], ' lf1    : ', ACC_lf_correct[1][i], ' / ', ACC_lf_total[1], ' -> ', 100.*ACC_lf_correct[1][i]/ACC_lf_total[1])
        print(condiction_name[i], ' lf2    : ', ACC_lf_correct[2][i], ' / ', ACC_lf_total[2], ' -> ', 100.*ACC_lf_correct[2][i]/ACC_lf_total[2])
        print()
        


# Read file
sents, sents_len = tokenize_dialogue(data_file)
kb_sents, _ = tokenize_kb(kb_file)
predict_query, predict_gate = tokenize_query(p_query_file)
g_query, g_gate = tokenize_query(g_query_file)

# sort the data order
sort_indices = sorted(range(len(sents_len)), key=lambda k: sents_len[k], reverse=True)
sort_indices_reverse = sorted(range(len(sort_indices)), key=lambda k: sort_indices[k])
sort_predict_query, sort_predict_gate = [], []
sort_g_query, sort_g_gate = [], []
for i in range(len(predict_query)):
    sort_predict_query.append(predict_query[sort_indices_reverse[i]])
    sort_predict_gate.append(predict_gate[sort_indices_reverse[i]])
    sort_g_query.append(g_query[sort_indices_reverse[i]])
    sort_g_gate.append(g_gate[sort_indices_reverse[i]])

# calculate accuracy
simulate_DB(sents, kb_sents, sort_predict_query, sort_predict_gate, sort_g_query, sort_g_gate, 13)
print('End')

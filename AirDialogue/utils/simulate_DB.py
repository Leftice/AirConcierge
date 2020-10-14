import pickle
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--n_sample', type=int, default=-1, help='N_h')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
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
    # data + kb
    kb_file =  data_path2 + 'tokenized/dev/dev.eval.kb'
    data_file = data_path2 + 'tokenized/dev/dev.eval.data'
    # eval step sql
    query_file = data_path + 'dev_sql/dev_predict_query'
    query2_file = data_path + 'dev_sql/dev_simple' 
    true_query_file = data_path + 'dev_sql/dev_gt_query'
    gate_file = data_path + 'dev_sql/dev_gate'
    # output file
    if not os.path.exists(data_path + 'dev_sql/simulate_DB/'):
        os.makedirs(data_path + 'dev_sql/simulate_DB/')
    small_fp = open(data_path + 'dev_sql/simulate_DB/small_db.kb', 'w')
    r_fp = open(data_path + 'dev_sql/simulate_DB/record', 'w')
    rf_fp = open(data_path + 'dev_sql/simulate_DB/filtered_kb', 'w')

elif args.train:

    kb_file = data_path2 + 'tokenized/train/train.kb'
    data_file = data_path2 + 'tokenized/train/train.data'

    query_file = data_path + 'train_sql/train_predict_query'
    query2_file = data_path + 'train_sql/train_simple' 
    true_query_file = data_path + 'train_sql/train_gt_query'
    gate_file = data_path + 'train_sql/train_gate'

    if not os.path.exists(data_path + 'train_sql/simulate_DB/'):
        os.makedirs(data_path + 'train_sql/simulate_DB/')
    small_fp = open(data_path + 'train_sql/simulate_DB/small_db.kb', 'w')
    r_fp = open(data_path + 'train_sql/simulate_DB/record', 'w')
    rf_fp = open(data_path + 'train_sql/simulate_DB/filtered_kb', 'w')

else:

    print('Please use --dev or --train !')
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
        for line in f:
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
    with open(path, 'r') as f:
        for line in f:
            words = []
            for word in line[38:-1].split('AND'): # without SELECT and \n
                words.append(word)
            query.append(words)
    return query

def tokenize_true_query(path):
    query = []
    truth_gate = []
    with open(path, 'r') as f:
        for line in f:
            words = []
            truth_gate.append(int(line[0]))
            for word in line[42:-1].split('AND'): # without SELECT and \n
                words.append(word)
            query.append(words)
    return query, truth_gate

def tokenize_query2(path):
    query = []
    with open(path, 'r') as f:
        for line in f:
            words = []
            for word in line.split(): # without SELECT and \n
                if word != '0.0':
                    words.append(int(word))
                else:
                    words.append(word)
            query.append(words)
    return query

def tokenize_gate(path):
    gate = []
    with open(path, 'r') as f:
        for line in f:
            words = []
            for word in line.split(): # without SELECT and \n
                words.append(word)
            gate.append(word)
    return gate

def translate_query_to_simple(query):
    condiction = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', \
                 'price', 'num_connections', 'airline_preference']

    simple_query = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    curr = 0
    for i in range(12):
        if curr == len(query):
            break
        if condiction[i] in query[curr]:
            simple_query[i] = int(query[curr].split()[-1])
            curr += 1
    return simple_query

def ACC_ex(true_flight, each_flight, each_flight_truth, ACC_ex_correct):
    if len(each_flight_truth) == 0 and len(each_flight) == 0:
        ACC_ex_correct += 1
    elif each_flight_truth == each_flight:
        ACC_ex_correct += 1
    return ACC_ex_correct

def ACC_lf(truth_gate, gate, true_query, query2, ACC_lf_correct, ACC_lf_total):
    
    if truth_gate == 1 and gate == '0.0':
        for i in range(12):
            if true_query[i] != -1:
                ACC_lf_total[i] += 1
    if truth_gate == 1 and gate == '1.0':
        # 4 combination : truth(y, y, y, n) predict(w, r, n, y)
        for i in range(12):
            if true_query[i] != -1: # y
                if (query2[i] != true_query[i]) and query2[i] != -1: # w
                    ACC_lf_total[i] += 1
                if query2[i] == true_query[i]: # r
                    ACC_lf_correct[i] += 1
                    ACC_lf_total[i] += 1
                if query2[i] == -1: # n
                    ACC_lf_total[i] += 1
            if true_query[i] == -1: # n
                if query2[i] != -1: # y
                    ACC_lf_total[i] += 1
    # if truth_gate == 0 and gate[i] == '0.0':
    if truth_gate == 0 and gate == '1.0':
        for i in range(12):
            if query2[i] != -1:
                ACC_lf_total[i] += 1
    return ACC_lf_correct, ACC_lf_total

def ACC_lf2(truth_gate, gate, true_query, query2, ACC_lf_correct, ACC_lf_total):
    
    if truth_gate == 1 and gate == '0.0':
        ACC_lf_total += 1
    if truth_gate == 1 and gate == '1.0':
        for i in range(1,13):
            if true_query[:i] == query2[:i]:
               ACC_lf_correct[i-1] += 1 
        ACC_lf_total += 1
    # if truth_gate == 0 and gate[i] == '0.0':
    if truth_gate == 0 and gate == '1.0':
        ACC_lf_total += 1
    return ACC_lf_correct, ACC_lf_total

def simulate_DB(kb, true_query, truth_gate, query, query2, gate, condiction_num, sort_indices, sort_sent):

    # <a1_MCO> <a2_LGA> <m1_Feb> <m2_Feb> <d1_9> <d2_10> <tn1_2> <tn2_7> <cl_business> <pr_400> <cn_1> <al_Southwest> <fl_1000>
    airport_list = ['DEN', 'LAX', 'MSP', 'DFW', 'SEA', 'ATL', 'IAH', 'DTW', 'ORD', 'IAD', 'CLT', 'EWR', 'LGA', 'JFK', 'HOU', 'SFO', 'AUS', 'OAK', 'LAS', 'PHL', 'BOS', 'MCO', 'DCA', 'PHX']
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    time_list = ['morning', 'afternoon', 'evening']
    air_class_list = ['economy', 'business']
    max_price_list = ['200', '500', '1000', '5000']
    airline_preference_list = ['normal-cost']

    ACC_ex_correct = 0
    ACC_ex_total = 0

    max_kb = 0
    record = []
    total = 0
    keep = 0
    error, error_truth = 0, 0

    ACC_lf_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ACC_lf_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ACC_lf_correct2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ACC_lf_total2 = 0

    ACC_f = [0, 0, 0, 0, 0, 0, 0]
    ACC_f_truth = [0, 0, 0, 0, 0, 0, 0]
    ACC_s = [0, 0, 0, 0, 0, 0, 0]
    ACC_total = [18460, 7599, 58, 132, 4487, 563, 4357]

    kb_len = [0 for _ in range(30)]

    filtered_kb = [0 for _ in range(30)]
    small_db = []
    samll_flight = []
    for i in range(len(kb)):

        true_query_i = translate_query_to_simple(true_query[i])
        ACC_lf_correct, ACC_lf_total = ACC_lf(truth_gate[i], gate[i], true_query_i, query2[i], ACC_lf_correct, ACC_lf_total)
        ACC_lf_correct2, ACC_lf_total2 = ACC_lf2(truth_gate[i], gate[i], true_query_i, query2[i], ACC_lf_correct2, ACC_lf_total2)

        if gate[i] == '0.0':
            small_db.append([sort_indices[i], kb[i][0:17]])
            record.append([sort_indices[i], 0])
            if truth_gate[i] == 0 and gate[i] == '0.0':
                ACC_f[4] += 1; ACC_f[5] += 1; ACC_f[6] += 1
            continue
        else:
            total += 30
        if truth_gate[i] == 1:
            ACC_ex_total += 1
        each_kb = []
        each_flight = []
        each_price = []
        each_flight_truth = []
        each_price_truth = []

        if i == 0:
            print('*'*100)
            print('query2 : ', query2[i], 'gate : ', gate[i], 'truth gate : ', truth_gate[i])
            print('air : ', airport_list[query2[i][0]], ' ', airport_list[query2[i][1]] ,'month : ', month_list[query2[i][2]], ' ', month_list[query2[i][3]])
            print('query : ', query[i])
            print('truth query : ', true_query[i])
            print('truth query : ', true_query_i)
            print('Our query   : ', query2[i])
        # our sql
        for entry in range(len(kb[i])):
            if i == 0:
                print('kb[i][entry] : ', kb[i][entry])
            correct = 1
            for c in range(len(kb[i][entry])-1) : # without flight number
                if query2[i][c] == -1:
                    continue 
                if c >= condiction_num:
                    break
                token = kb[i][entry][c].split('_', 1)[1].split('>', 1)[0]
                # print('query2 : ', query2[i][c], 'c : ',  c)
                if c == 0:
                    airport_index = airport_list.index(token)
                    if airport_index != query2[i][c]:
                        correct = 0
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('airport_index : ', airport_index)
                        # raise
                        break
                elif c == 1:
                    airport_index = airport_list.index(token)
                    if airport_index != query2[i][c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('airport_index : ', airport_index)
                        # raise
                        correct = 0
                        break
                elif c == 2:
                    month_index = month_list.index(token)
                    if month_index != query2[i][c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('month_index : ', month_index)
                        # raise
                        correct = 0
                        break
                elif c == 3:
                    month_index = month_list.index(token)
                    if month_index != query2[i][c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('month_index : ', month_index)
                        # raise
                        correct = 0
                        break
                elif c == 4:
                    if int(token)-1 != query2[i][c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('day_index : ', int(token)-1)
                        # raise
                        correct = 0
                        break
                elif c == 5:
                    if int(token)-1 != query2[i][c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('day_index : ', int(token)-1)
                        # raise
                        correct = 0
                        break
                elif c == 6:
                    d_time = time_list[query2[i][c]]
                    if d_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        correct = 0
                    if d_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                        correct = 0
                    if d_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                        correct = 0
                elif c == 7:
                    r_time = time_list[query2[i][c]]
                    if r_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        correct = 0
                    if r_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                        correct = 0
                    if r_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                        correct = 0
                elif c == 8:
                    class_index = air_class_list.index(token)
                    if class_index != query2[i][c]:
                        correct = 0
                elif c == 9:
                    if int(max_price_list[query2[i][c]]) < int(token):
                        correct = 0
                elif c == 10:
                    if true_query_i[c] < int(token):
                        correct = 0
                elif c == 11:
                    if true_query_i[c] == 1 and token not in ['UA', 'AA', 'Delta', 'Hawaiian']:
                        correct = 0
            if correct == 1:
                each_price.append(int(kb[i][entry][9].split('_', 1)[1].split('>', 1)[0]))
                each_flight.append(kb[i][entry][12].split('_', 1)[1].split('>', 1)[0])
                each_kb.append(kb[i][entry])
                keep += 1
        # ground truth
        for entry in range(len(kb[i])):
            correct = 1
            for c in range(len(kb[i][entry])-1):
                if true_query_i[c] == -1:
                    continue 
                if c >= condiction_num:
                    break
                token = kb[i][entry][c].split('_', 1)[1].split('>', 1)[0]
                # print('query2 : ', query2[i][c], 'c : ',  c)
                if c == 0:
                    airport_index = airport_list.index(token)
                    if airport_index != true_query_i[c]:
                        correct = 0
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('airport_index : ', airport_index)
                        # raise
                        break
                elif c == 1:
                    airport_index = airport_list.index(token)
                    if airport_index != true_query_i[c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('airport_index : ', airport_index)
                        # raise
                        correct = 0
                        break
                elif c == 2:
                    month_index = month_list.index(token)
                    if month_index != true_query_i[c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('month_index : ', month_index)
                        # raise
                        correct = 0
                        break
                elif c == 3:
                    month_index = month_list.index(token)
                    if month_index != true_query_i[c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('month_index : ', month_index)
                        # raise
                        correct = 0
                        break
                elif c == 4:
                    if int(token)-1 != true_query_i[c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('day_index : ', int(token)-1)
                        # raise
                        correct = 0
                        break
                elif c == 5:
                    if int(token)-1 != true_query_i[c]:
                        # print('kb[i][entry] : ', kb[i][entry])
                        # print('query2[i][c] : ', query2[i][c])
                        # print('day_index : ', int(token)-1)
                        # raise
                        correct = 0
                        break
                elif c == 6:
                    d_time = time_list[true_query_i[c]]
                    if d_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        correct = 0
                    if d_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                        correct = 0
                    if d_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                        correct = 0
                elif c == 7:
                    r_time = time_list[true_query_i[c]]
                    if r_time == 'morning' and int(token) not in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        correct = 0
                    if r_time == 'afternoon' and int(token) not in [12, 13, 14, 15, 16, 17, 18, 19]:
                        correct = 0
                    if r_time == 'evening' and int(token) not in [20, 21, 22, 23, 0, 1, 2]:
                        correct = 0
                elif c == 8:
                    class_index = air_class_list.index(token)
                    if class_index != true_query_i[c]:
                        correct = 0
                elif c == 9:
                    if int(max_price_list[true_query_i[c]]) < int(token):
                        correct = 0
                elif c == 10:
                    if true_query_i[c] < int(token):
                        correct = 0
                elif c == 11:
                    if true_query_i[c] == 1 and token not in ['UA', 'AA', 'Delta', 'Hawaiian']:
                        correct = 0
            if correct == 1:
                each_price_truth.append(int(kb[i][entry][9].split('_', 1)[1].split('>', 1)[0]))
                each_flight_truth.append(kb[i][entry][12].split('_', 1)[1].split('>', 1)[0])
        action = sort_sent[i][1]
        if len(action) != 4:
            print('No name ! ', sort_indices[i])
            true_flight = action[0].split('_', 1)[1].split('>', 1)[0]
            raise
        else:
            true_flight = action[2].split('_', 1)[1].split('>', 1)[0]
            if i == 0:
                print('true_flight : ', true_flight)
        ACC_ex_correct = ACC_ex(true_flight, each_flight, each_flight_truth, ACC_ex_correct)

        # sort price
        index_price = sorted(range(len(each_price)), key=lambda k: each_price[k])
        each_flight = [each_flight[p] for p in index_price ]
        index_price_truth = sorted(range(len(each_price_truth)), key=lambda k: each_price_truth[k])
        each_flight_truth = [each_flight_truth[p] for p in index_price_truth ]

        # empty : change-no_flight, book--no_flight, cancel-no_reservation, change-no_reservation, cancel-cancel
        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight == 'empty' and len(each_flight) == 0:
            # book--no_flight, change-no_flight
            ACC_f[1] += 1 # book--no_flight
            ACC_f[3] += 1 # change-no_flight
        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight != 'empty' and len(each_flight) == 1 and (true_flight in each_flight):
            # book--no_flight, change-no_flight
            ACC_f[0] += 1 # book--book
            ACC_f[2] += 1 # change-book
        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight != 'empty' and len(each_flight) > 1 and (int(true_flight) == int(each_flight[0])):
            # book--no_flight, change-no_flight
            ACC_f[0] += 1 # book--book
            ACC_f[2] += 1 # change-book

        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight == 'empty' and len(each_flight_truth) == 0:
            # book--no_flight, change-no_flight
            ACC_f_truth[1] += 1 # book--no_flight
            ACC_f_truth[3] += 1 # change-no_flight
        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight != 'empty' and len(each_flight_truth) == 1 and (true_flight in each_flight_truth):
            # book--no_flight, change-no_flight
            ACC_f_truth[0] += 1 # book--book
            ACC_f_truth[2] += 1 # change-book
        if truth_gate[i] == 1 and gate[i] != '0.0' and true_flight != 'empty' and len(each_flight_truth) > 1 and (int(true_flight) == int(each_flight_truth[0])):
            # book--no_flight, change-no_flight
            ACC_f_truth[0] += 1 # book--book
            ACC_f_truth[2] += 1 # change-book

        if true_flight == 'empty' and len(each_flight) == 0:
            # print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight, 'True Empty')
            record.append([sort_indices[i], each_flight, true_flight])
        elif true_flight == 'empty' and len(each_flight) != 0:
            # print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight)
            record.append([sort_indices[i], each_flight, true_flight])
        elif true_flight in each_flight:
            # print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight, 'True Flight')
            record.append([sort_indices[i], each_flight, true_flight])
        elif true_flight != 'empty' and true_flight not in each_flight_truth:
            # print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight, 'Error Flight')
            # print('T : ', true_query[i])
            # print('P : ', query[i])
            record.append([sort_indices[i], each_flight, true_flight, true_query[i], query[i]])
            print('*'*100)
            print('Sample : ', i)
            print('each_flight_truth : ', each_flight_truth)
            print('true_flight : ', true_flight)
            print('query2 : ', query2[i], 'gate : ', gate[i], 'truth gate : ', truth_gate[i])
            print('token : ', airport_list[query2[i][0]], ' ', airport_list[query2[i][1]] , month_list[query2[i][2]], ' ', month_list[query2[i][3]])
            print('query : ', query[i])
            print('truth query : ', true_query[i])
            print('truth query : ', true_query_i)
            print('Our query   : ', query2[i])
            for k in range(30):
                print('kb[i][entry] : ', kb[i][k])
            print('sents : ', sort_sent[i])
            print('*'*100)
            
            error_truth += 1

        if true_flight != 'empty' and true_flight not in each_flight:
            # print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight, 'Error Flight')
            # print('T : ', true_query[i])
            # print('P : ', query[i])
            record.append([sort_indices[i], each_flight, true_flight, true_query[i], query[i]])            
            error += 1
            
        kb_len[len(each_kb)] += 1
        if len(each_kb) > max_kb:
            max_kb = len(each_kb)
        while len(each_kb) < 17:
            each_kb.append(kb[i][-1])
        # print('each_kb : ', each_kb)
        # raise

        # if sort_indices[i] == 25544:
        #     print('sample : ', sort_indices[i], ' flight : ', each_flight, 'flight : ', true_flight, 'Error Flight')
        #     print(each_kb)
        #     print(len(each_kb))

        small_db.append([sort_indices[i], each_kb])
        samll_flight.append(each_flight)

    print('Max kb : ', max_kb)
    print('Acc ex : ', 100.*ACC_ex_correct/ACC_ex_total, ACC_ex_correct, ACC_ex_total)
    condiction_name = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', \
                 'price', 'num_connections', 'airline_preference']
    for i in range(12):
        print(condiction_name[i], ' : ', ACC_lf_correct[i], ' / ', ACC_lf_total[i], ' -> ', 100.*ACC_lf_correct[i]/ACC_lf_total[i])
        print(condiction_name[i], ' : ', ACC_lf_correct2[i], ' / ', ACC_lf_total2, ' -> ', 100.*ACC_lf_correct2[i]/ACC_lf_total2)
    print('ACC_Flight : ', ACC_f)
    print('ACC_Flight_truth : ', ACC_f_truth)

    print('kb_len : ', kb_len)
    return samll_flight, small_db, total, keep, error, error_truth, record

sents, sents_len = tokenize_dialogue(data_file)
kb_sents, reservations = tokenize_kb(kb_file)

if args.n_sample != -1:
    sents = sents[:args.n_sample]
    sents_len = sents_len[:args.n_sample]
    kb_sents = kb_sents[:args.n_sample]
    reservations = reservations[:args.n_sample]

print('Size of kb : ', len(kb_sents))
print('Size of each kb : ', len(kb_sents[0]), kb_sents[0][0])

query = tokenize_query(query_file)
print('Size of query : ', len(query))
print('Size of each query : ', len(query[0]), query[0])

true_query, truth_gate = tokenize_true_query(true_query_file)
print('Size of true_query : ', len(true_query))
print('Size of each true_query : ', len(true_query[0]), true_query[0])

query2 = tokenize_query2(query2_file)
print('Size of query2 : ', len(query2))
print('Size of each query2 : ', len(query2[0]), query2[0])

gate = tokenize_gate(gate_file)
print('Size of gate : ', len(gate))
print('Size of each gate : ', gate[0])

sort_indices = sorted(range(len(sents_len)), key=lambda k: sents_len[k], reverse=True)
sort_indices_reverse = sorted(range(len(sort_indices)), key=lambda k: sort_indices[k])

sort_true_query, sort_truth_gate, sort_query, sort_query2, sort_gate = [], [], [], [], []
for i in range(len(true_query)):
    sort_true_query.append(true_query[sort_indices_reverse[i]])
    sort_truth_gate.append(truth_gate[sort_indices_reverse[i]])
    sort_query.append(query[sort_indices_reverse[i]])
    sort_query2.append(query2[sort_indices_reverse[i]])
    sort_gate.append(gate[sort_indices_reverse[i]])

sort_kb = []
sort_sent = []
sort_reservations = []
for i in range(len(kb_sents)):
    sort_kb.append(kb_sents[sort_indices[i]])
    sort_sent.append(sents[sort_indices[i]])
    sort_reservations.append(reservations[sort_indices[i]])
print('*'*100)
# print('sort_indices : ', sort_indices[0:2])
# print('kb_sents : ', sort_kb[0:2])
# print('sents : ', sort_sent[0:2])
print('indices : ', sort_indices_reverse[0:2])
print('kb_sents : ', kb_sents[0:2])
print('sents : ', sents[0:2])

# samll_flight, small_db, total, keep, error, record = simulate_DB(sort_kb, true_query, truth_gate, query, query2, gate, 6, sort_indices, sort_sent)
samll_flight, small_db, total, keep, error, error_truth, record = simulate_DB(kb_sents, sort_true_query, sort_truth_gate, sort_query, sort_query2, sort_gate, 12, list(range(len(kb_sents))), sents)

print('keep : ', keep)
print('total : ', total)
print('error : ', error, ' / ', len(kb_sents))
print('error_truth : ', error_truth, ' / ', len(kb_sents))
print('record : ', len(record))

record.sort(key=lambda x: x[0])
for i in range(len(record)):
    r_fp.write(str(record[i]) + '\n')

small_db.sort(key=lambda x: x[0])
for i in range(len(small_db)):
    words = str(reservations[i]) + ' '
    for entry in range(len(small_db[i][1])):
        for word in small_db[i][1][entry]:
            words += str(word) + ' '
    small_fp.write(words)
    small_fp.write('\n')

print('End')
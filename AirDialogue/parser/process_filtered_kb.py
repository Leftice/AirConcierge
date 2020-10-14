import json
import argparse
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--self_play_eval', action='store_true')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
args = parser.parse_args()

if args.syn:
    sql_path = './synthesized_sql/'
    data_path = '../data/synthesized/'
elif args.air:
    sql_path = './airdialogue_sql/'
    data_path = '../data/airdialogue/'
else:
    print('Pleae use --syn or --air !')
    raise

if args.train:

    kb_file = data_path + 'tokenized/train/train.kb'
    data_file = data_path + 'tokenized/train/train.data'
    
    stk_path = data_path + 'SQL/train/State_Tracking.txt'

    true_query_file = sql_path + 'train.sql.txt'
    rf_fp = open(data_path + 'SQL/train/filtered_kb', 'w')
    log_fp = open(sql_path + 'train.filtered.log', 'w')

elif args.dev:

    kb_file = data_path + 'tokenized/dev/dev.eval.kb'
    data_file = data_path + 'tokenized/dev/dev.eval.data'
    
    stk_path = data_path + 'SQL/dev/State_Tracking.txt'

    true_query_file =  sql_path + 'dev.sql.txt'
    rf_fp = open(data_path + 'SQL/dev/filtered_kb', 'w')
    log_fp = open(sql_path + 'dev.filtered.log', 'w')

else:
    print('Pleae use --train or --dev !')
    raise

def tokenize_query(path):
    query = []
    with open(path, 'r') as f:
        for line in f:
            words = []
            line = line[0:-1].replace('\'', '')
            for word in line[38:].split('AND'): # without SELECT and \n
                words.append(word)
            query.append(words)
    return query

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

def read_SQL_YN(path):
    with open(path, 'r') as f:
        all_SQL_YN = []
        for line in f:
            line = line.replace(',', '').replace('\'', '')
            items = line.split("|")

            SQL_YN = 1
            if items[2].split()[0] == 'Y_SQL':
                SQL_YN = 1
            elif items[2].split()[0] == 'N_SQL':
                SQL_YN = 0
            else:
                print('SQL ERROR !')
                raise
            all_SQL_YN.append(SQL_YN)

    return all_SQL_YN

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

def simulate_DB(sents, kb, truth_gate, true_query, condiction_num):

    # <a1_MCO> <a2_LGA> <m1_Feb> <m2_Feb> <d1_9> <d2_10> <tn1_2> <tn2_7> <cl_business> <pr_400> <cn_1> <al_Southwest> <fl_1000>
    airport_list = ['DEN', 'LAX', 'MSP', 'DFW', 'SEA', 'ATL', 'IAH', 'DTW', 'ORD', 'IAD', 'CLT', 'EWR', 'LGA', 'JFK', 'HOU', 'SFO', 'AUS', 'OAK', 'LAS', 'PHL', 'BOS', 'MCO', 'DCA', 'PHX']
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    time_list = ['morning', 'afternoon', 'evening']
    air_class_list = ['economy', 'business']
    max_price_list = ['200', '500', '1000', '5000']
    airline_preference_list = ['normal-cost']

    max_kb = 0
    total = 0
    error = 0

    ACC_f_truth = [0, 0, 0, 0, 0, 0, 0]
    ACC_total = [18460, 7599, 58, 132, 4487, 563, 4357]

    kb_len = [0 for _ in range(30)]

    for i in tqdm(range(len(kb))):

        filtered_kb = [0 for _ in range(30)]
        if truth_gate[i] == 0:
            for w in range(30):
                rf_fp.write(str(filtered_kb[w])+' ')
            rf_fp.write('| '+ str(truth_gate[i]) + ' ')
            rf_fp.write('| 0 ')
            rf_fp.write('\n')
            continue

        true_query_i = translate_query_to_simple(true_query[i])

        each_flight_truth = []
        each_price_truth = []
        each_kb_truth = []

        if i > 0 and i < 5:
            log_fp.write('*'*100 + '\n')
            log_fp.write('truth gate : ' + str(truth_gate[i]) + '\n')
            log_fp.write('air : ' +  str(airport_list[true_query_i[0]]) + ' ' + str(airport_list[true_query_i[1]]) + 'month : ' + str(month_list[true_query_i[2]]) + ' ' + str(month_list[true_query_i[3]]) + '\n')
            log_fp.write('truth query : ' + str(true_query[i]) + '\n')
            log_fp.write('truth query : ' + str(true_query_i) + '\n')

        # ground truth
        for entry in range(len(kb[i])):
            
            correct = 1
            for c in range(len(kb[i][entry])):
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
                        break
                elif c == 1:
                    airport_index = airport_list.index(token)
                    if airport_index != true_query_i[c]:
                        correct = 0
                        break
                elif c == 2:
                    month_index = month_list.index(token)
                    if month_index != true_query_i[c]:
                        correct = 0
                        break
                elif c == 3:
                    month_index = month_list.index(token)
                    if month_index != true_query_i[c]:
                        correct = 0
                        break
                elif c == 4:
                    if int(token)-1 != true_query_i[c]:
                        correct = 0
                        break
                elif c == 5:
                    if int(token)-1 != true_query_i[c]:
                        correct = 0
                        break
            if correct == 1:
                each_price_truth.append(int(kb[i][entry][9].split('_', 1)[1].split('>', 1)[0]))
                each_flight_truth.append(kb[i][entry][12].split('_', 1)[1].split('>', 1)[0])
                each_kb_truth.append(kb[i][entry])
                filtered_kb[entry] = 1
            if i > 0 and i < 5:
                log_fp.write(str(filtered_kb[entry]) + ' kb[i][entry] : ' + str(kb[i][entry]) + '\n')

        for w in range(30):
            rf_fp.write(str(filtered_kb[w])+' ')
        rf_fp.write('| '+ str(truth_gate[i]) + ' ')
        rf_fp.write('| '+ str(len(each_kb_truth)) + ' ')
        rf_fp.write('\n')

        action = sents[i][1]
        if len(action) != 4:
            print('No name ! ', sort_indices[i])
            true_flight = action[0].split('_', 1)[1].split('>', 1)[0]
            raise
        else:
            true_flight = action[2].split('_', 1)[1].split('>', 1)[0]
            if i > 1 and i < 5:
                log_fp.write('true_flight : ' + str(true_flight) + '\n')

        # sort price
        index_price_truth = sorted(range(len(each_price_truth)), key=lambda k: each_price_truth[k])
        each_flight_truth = [each_flight_truth[p] for p in index_price_truth ]

        if truth_gate[i] == 1 and true_flight == 'empty' and len(each_flight_truth) == 0:
            # book--no_flight, change-no_flight
            ACC_f_truth[1] += 1 # book--no_flight
            ACC_f_truth[3] += 1 # change-no_flight
        if truth_gate[i] == 1 and true_flight != 'empty' and len(each_flight_truth) == 1 and (true_flight in each_flight_truth):
            # book--no_flight, change-no_flight
            ACC_f_truth[0] += 1 # book--book
            ACC_f_truth[2] += 1 # change-book
        if truth_gate[i] == 1 and true_flight != 'empty' and len(each_flight_truth) > 1 and (int(true_flight) == int(each_flight_truth[0])):
            # book--no_flight, change-no_flight
            ACC_f_truth[0] += 1 # book--book
            ACC_f_truth[2] += 1 # change-book

        if true_flight != 'empty' and true_flight not in each_flight_truth:
            print('*'*100)
            print('Sample : ', i)
            print('each_flight_truth : ', each_flight_truth)
            print('true_flight : ', true_flight)
            print('truth gate : ', truth_gate[i])
            print('token : ', airport_list[query2[i][0]], ' ', airport_list[query2[i][1]] , month_list[query2[i][2]], ' ', month_list[query2[i][3]])
            print('query : ', query[i])
            print('truth query : ', true_query[i])
            print('truth query : ', true_query_i)
            print('Our query   : ', query2[i])
            for k in range(30):
                print('kb[i][entry] : ', kb[i][k])
            print('sents : ', sents[i])
            print('*'*100)
            error_truth += 1
            
        kb_len[len(each_kb_truth)] += 1
        if len(each_kb_truth) > max_kb:
            max_kb = len(each_kb_truth)

    log_fp.write('='*100 + '\n')
    log_fp.write('Max kb : ' + str(max_kb) + '\n')
    log_fp.write('ACC_Flight_truth : ' + str(ACC_f_truth) + '\n')
    log_fp.write('kb_len : ' + str(kb_len) + '\n')
    log_fp.write('error_truth : ' + str(error) + ' / ' + str(len(kb_sents)) + '\n')

print('Start processing filtered_kb !')
print('Reading data ...')
sents, sents_len = tokenize_dialogue(data_file)

print('Reading kb ...')
kb_sents, reservations = tokenize_kb(kb_file)

print('Reading SQL_YN ...')
SQL_YN = read_SQL_YN(stk_path)

print('Reading train.sql.txt ...')
true_query = tokenize_query(true_query_file)
log_fp.write('Size of true_query : ' + str(len(true_query)) + '\n')
log_fp.write('Size of each true_query : ' + str(len(true_query[0])) + ' ' +  str(true_query[0]) + '\n')

print('Processing ...')
simulate_DB(sents, kb_sents, SQL_YN, true_query, 6)

print('Finish processing filtered_kb !')
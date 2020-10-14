import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--self_play_dev', action='store_true', help='Use dev')
parser.add_argument('--infer_dev', action='store_true', help='Use dev')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
parser.add_argument('--toy', action='store_true', help='Use toy')
args = parser.parse_args()

if args.syn:
    data_path = './results/synthesized/'
    kb_path = './data/synthesized/'
elif args.air:
    data_path = './results/airdialogue/'
    kb_path = './data/airdialogue/'
else:
    print('Pleae use --syn or --air !')
    raise

if args.self_play_dev:
    if args.toy :
        kb_file =  kb_path + 'tokenized/sub_air/toy_dev.selfplay.eval.kb'
    else:
        kb_file =  kb_path + 'tokenized/selfplay_eval/dev.selfplay.eval.kb' 
    predict_query_file = data_path + 'SelfPlay_Eval/SQL/predicted_.txt'
    # true_query_file = 'Self_Play_Eval/SQL/ground_truth_.txt'
    if not os.path.exists(data_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/'):
        os.makedirs(data_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/')
    filtered_fp = open(data_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered_index.kb', 'w')
    filtered_fp2 = open(data_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered.kb', 'w')
    log_fp = open(data_path + 'SelfPlay_Eval/self_play_dev.log', 'w')
elif args.infer_dev:
    if args.toy : 
        kb_file =  kb_path + 'tokenized/sub_air/toy_dev.infer.kb'
    else:
        kb_file =  kb_path + 'tokenized/infer/dev.infer.kb'
    predict_query_file = data_path + 'Inference_Bleu/SQL/predicted_.txt'
    if not os.path.exists(data_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/'):
        os.makedirs(data_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/')
    filtered_fp = open(data_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered_index.kb', 'w')
    filtered_fp2 = open(data_path + 'Inference_Bleu/SQL/dev_sql/simulate_DB/filtered.kb', 'w')
    log_fp = open(data_path + 'Inference_Bleu/infer_dev.log', 'w')
else:
    print('Please use --dev !')
    raise

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
    gate = []
    with open(path, 'r') as f:
        for line in f:
            items = line.split("|")
            gate.append(int(items[0]))
            words = []
            for word in items[1][38:-1].split('AND'): # without SELECT and \n
                words.append(word)
            query.append(words)
    return query, gate

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

def simulate_DB(kb, query, gate, condiction_num, reservations):

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
    
    filtered_kb = []
    kb_len = [0 for _ in range(30)]

    for i in tqdm(range(len(kb))):
        # print('id : ', i)

        filtered_index = [0 for _ in range(30)]
        if gate[i] == 0:
            for w in range(30):
                filtered_fp.write(str(filtered_index[w])+' ')
            filtered_fp.write('| '+ str(gate[i]) + '\n')
            words = str(reservations[i]) + ' ' + '<st_no_reservation_cancel> '
            filtered_fp2.write(words + '\n')
            continue

        query_i = translate_query_to_simple(query[i])
        each_kb = []
        each_flight = []

        if i == 0:
            log_fp.write('*'*100 + '\n')
            log_fp.write('query : ' + str(query[i]) + 'gate : ' +  str(gate[i]) + '\n')
            log_fp.write('air : ' + str(airport_list[query_i[0]]) + ' ' + str(airport_list[query_i[1]]) + 'month : ' + str(month_list[query_i[2]]) + ' ' + str(month_list[query_i[3]]) + '\n')

        # our sql
        for entry in range(len(kb[i])):
            correct = 1
            for c in range(len(kb[i][entry])):
                if query_i[c] == -1:
                    continue 
                if c >= condiction_num:
                    break
                token = kb[i][entry][c].split('_', 1)[1].split('>', 1)[0]
                if c == 0:
                    airport_index = airport_list.index(token)
                    if airport_index != query_i[c]:
                        correct = 0
                        break
                elif c == 1:
                    airport_index = airport_list.index(token)
                    if airport_index != query_i[c]:
                        correct = 0
                        break
                elif c == 2:
                    month_index = month_list.index(token)
                    if month_index != query_i[c]:
                        correct = 0
                        break
                elif c == 3:
                    month_index = month_list.index(token)
                    if month_index != query_i[c]:
                        correct = 0
                        break
                elif c == 4:
                    if int(token)-1 != query_i[c]:
                        correct = 0
                        break
                elif c == 5:
                    if int(token)-1 != query_i[c]:
                        correct = 0
                        break
            if correct == 1:
                each_kb.append(kb[i][entry])
                each_flight.append(kb[i][entry][12].split('_', 1)[1].split('>', 1)[0])
                filtered_index[entry] = 1
            
        for w in range(30):
            filtered_fp.write(str(filtered_index[w])+' ')
        filtered_fp.write('| '+ str(gate[i]) + '\n')

        if i == 0:
            log_fp.write('filtered_index : ' + str(filtered_index) + '\n')
            log_fp.write('each_flight : ' + str(each_flight) + '\n')
            for k in range(30):
                log_fp.write('kb : ' + str(kb[i][k]) + '\n')
            log_fp.write('*'*100 + '\n')

        kb_len[len(each_kb)] += 1
        if len(each_kb) > max_kb:
            max_kb = len(each_kb)

        filtered_kb.append(each_kb)

        words = str(reservations[i]) + ' '
        if len(each_kb) == 0:
            words += '<fl_empty> '
        for entry in range(len(each_kb)):
            for word in each_kb[entry]:
                words += str(word) + ' '
        filtered_fp2.write(words + '\n')

    log_fp.write('Max kb : ' + str(max_kb) + '\n')
    log_fp.write('kb_len : ' + str(kb_len) + '\n')

kb_sents, reservations = tokenize_kb(kb_file)

log_fp.write('Size of kb : ' + str(len(kb_sents)) + '\n')
log_fp.write('Size of each kb : ' + str(len(kb_sents[0])) + ' ' + str(kb_sents[0][0]) + '\n')
predict_query, predict_gate = tokenize_query(predict_query_file)

log_fp.write('Size of query : ' + str(len(predict_query)))
log_fp.write('Size of each query : ' + str(len(predict_query[0])) + ' ' + str(predict_query[0]) + '\n')
log_fp.write('Size of each gate  : ' + str(predict_gate[0]) + '\n')
simulate_DB(kb_sents, predict_query, predict_gate, 6, reservations)

# truth_query, truth_gate = tokenize_query(true_query_file)
# print('Size of true_query : ', len(truth_query))
# print('Size of each true_query : ', len(truth_query[0]), truth_query[0])
# print('Size of each gate : ', truth_gate[0])
# filtered_kb, filiter_indexes = simulate_DB(kb_sents, truth_query, truth_gate, 6, reservations)

print('End')
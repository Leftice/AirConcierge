from tqdm import tqdm 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')
args = parser.parse_args()

if args.syn:
    if not os.path.exists('./synthesized_label/'):
        os.makedirs('./synthesized_label/')
    label_path = './synthesized_label/'
    data_path = '../data/synthesized/'
elif args.air:
    if not os.path.exists('./airdialogue_label/'):
        os.makedirs('./airdialogue_label/')
    label_path = './airdialogue_label/'
    data_path = '../data/airdialogue/'
else:
    print('Pleae use --syn or --air !')
    raise

# SQL line, read dialogue line, read kb line
# sql_fp = open('train.sql.txt', 'w')
# data_fp = open('/home/sclab/pytorch-seq2seq/data/sub_air/train_sub.data', "r")
# kb_fp = open('/home/sclab/pytorch-seq2seq/data/sub_air/train_sub.kb', "r")
if args.dev:
    if not os.path.exists(label_path + 'dev_label_result/'):
        os.makedirs(label_path + 'dev_label_result/')
    if not os.path.exists(label_path + 'dev_label_result_fail/'):
        os.makedirs(label_path + 'dev_label_result_fail/')
    data_fp = open(data_path + 'tokenized/dev/dev.eval.data', "r")
    kb_fp = open(data_path + 'tokenized/dev/dev.eval.kb', "r")
    state_tracking_fp = open(data_path + 'SQL/dev/State_Tracking.txt', 'w')
    log_fp = open(label_path + 'dev.status.log', 'w')

    bb_fp = open(label_path + 'dev_label_result/Book_Book.txt', 'w')
    bn_fp = open(label_path + 'dev_label_result/Book_No_flight.txt', 'w')
    chch_fp = open(label_path + 'dev_label_result/Change_Change.txt', 'w')
    chno_fp = open(label_path + 'dev_label_result/Change_No_flight.txt', 'w')
    chnor_fp = open(label_path + 'dev_label_result/Change_No_reservation.txt', 'w')
    cc_fp = open(label_path + 'dev_label_result/Cancel_Cancel.txt', 'w')
    cnor_fp = open(label_path + 'dev_label_result/Cancel_No_reservation.txt', 'w')
    error_report_fp = open(label_path + 'dev_Error.txt', 'w')
    fail_bb_fp = open(label_path + 'dev_label_result_fail/Book_Book_fail.txt', 'w')
    fail_bn_fp = open(label_path + 'dev_label_result_fail/Book_No_flight_fail.txt', 'w')
    fail_chch_fp = open(label_path + 'dev_label_result_fail/Change_Change_fail.txt', 'w')
    fail_chno_fp = open(label_path + 'dev_label_result_fail/Change_No_flight_fail.txt', 'w')
    fail_chnor_fp = open(label_path + 'dev_label_result_fail/Change_No_reservation_fail.txt', 'w')
    fail_cc_fp = open(label_path + 'dev_label_result_fail/Cancel_Cancel_fail.txt', 'w')
    fail_cnor_fp = open(label_path + 'dev_label_result_fail/Cancel_No_reservation_fail.txt', 'w')
elif args.train:
    if not os.path.exists(label_path + 'train_label_result/'):
        os.makedirs(label_path + 'train_label_result/')
    if not os.path.exists(label_path + 'train_label_result_fail/'):
        os.makedirs(label_path + 'train_label_result_fail/')
    data_fp = open(data_path + 'tokenized/train/train.data', "r")
    kb_fp = open(data_path + 'tokenized/train/train.kb', "r")
    state_tracking_fp = open(data_path + 'SQL/train/State_Tracking.txt', 'w')
    log_fp = open(label_path + 'train.status.log', 'w')

    bb_fp = open(label_path + 'train_label_result/Book_Book.txt', 'w')
    bn_fp = open(label_path + 'train_label_result/Book_No_flight.txt', 'w')
    chch_fp = open(label_path + 'train_label_result/Change_Change.txt', 'w')
    chno_fp = open(label_path + 'train_label_result/Change_No_flight.txt', 'w')
    chnor_fp = open(label_path + 'train_label_result/Change_No_reservation.txt', 'w')
    cc_fp = open(label_path + 'train_label_result/Cancel_Cancel.txt', 'w')
    cnor_fp = open(label_path + 'train_label_result/Cancel_No_reservation.txt', 'w')
    error_report_fp = open(label_path + 'train_Error.txt', 'w')
    fail_bb_fp = open(label_path + 'train_label_result_fail/Book_Book_fail.txt', 'w')
    fail_bn_fp = open(label_path + 'train_label_result_fail/Book_No_flight_fail.txt', 'w')
    fail_chch_fp = open(label_path + 'train_label_result_fail/Change_Change_fail.txt', 'w')
    fail_chno_fp = open(label_path + 'train_label_result_fail/Change_No_flight_fail.txt', 'w')
    fail_chnor_fp = open(label_path + 'train_label_result_fail/Change_No_reservation_fail.txt', 'w')
    fail_cc_fp = open(label_path + 'train_label_result_fail/Cancel_Cancel_fail.txt', 'w')
    fail_cnor_fp = open(label_path + 'train_label_result_fail/Cancel_No_reservation_fail.txt', 'w')
else:
    print('Pleae use --dev or --train !')
    raise

# intent :: dev.json
departure_airport_dict = {}
return_airport_dict = {}
departure_month_dict = {}
return_month_dict = {}
departure_day_dict = {}
return_day_dict = {}
departure_time_dict = {}
return_time_dict = {}
name1_dict = {}
name2_dict = {}
class_dict = {}
max_price_dict = {}
max_connections_dict = {}
airline_preference_dict = {}
goal_dict = {}

intent_name_list = ['departure_airport_dict', 'return_airport_dict', 'departure_month_dict', 'return_month_dict', 'departure_day_dict', 'return_day_dict',
               'departure_time_dict', 'return_time_dict', 'name1_dict', 'name2_dict', 'class_dict', 'max_price_dict', 'max_connections_dict',
               'airline_preference_dict', 'goal_dict']
intent_list = [departure_airport_dict, return_airport_dict, departure_month_dict, return_month_dict, departure_day_dict, return_day_dict,
               departure_time_dict, return_time_dict, name1_dict, name2_dict, class_dict, max_price_dict, max_connections_dict, 
               airline_preference_dict, goal_dict]

# state_name_combination = ['book--book', 'book--change', 'book--no_flight', 'book--cancel', 'book--no_reservation',
#                      'change--book', 'change--change', 'change--no_flight', 'change--cancel', 'change--no_reservation',
#                      'cancel--book', 'cancel--change', 'cancel--no_flight', 'cancel--cancel', 'cancel--no_reservation']
# state_combination = {'book--book':0, 'book--change':0, 'book--no_flight':0, 'book--cancel':0, 'book--no_reservation':0,
#                      'change--book':0, 'change--change':0, 'change--no_flight':0, 'change--cancel':0, 'change--no_reservation':0,
#                      'cancel--book':0, 'cancel--change':0, 'cancel--no_flight':0, 'cancel--cancel':0, 'cancel--no_reservation':0}
state_name_combination = ['book--book', 'book--no_flight',
                     'change--change', 'change--no_flight', 'change--no_reservation',
                     'cancel--cancel', 'cancel--no_reservation']
state_combination = {'book--book':0, 'book--no_flight':0,
                     'change--change':0, 'change--no_flight':0, 'change--no_reservation':0,
                     'cancel--cancel':0, 'cancel--no_reservation':0}

def split_index(index_list):
    list1 = []
    list2 = []
    pre = index_list[0]
    now = index_list[0]
    turn = -1
    for i in range(len(index_list)):
        now = index_list[i]
        if int(pre) > int(now):
            turn = i
            break
        else:
            pre = now  
    list1 = index_list[:turn]
    list2 = index_list[turn:]
    return list1, list2

def check_mark_index(index, dialogue_list, t2_index_list):
    mark_index = 0
    for i in range(len(dialogue_list)):
        if i == index:
            if str(mark_index) not in t2_index_list:
                print('*'*100)
                print('Error ! : ', index)
                print('t2_index_list : ', t2_index_list)
                print('mark_index : ', mark_index)
                print(dialogue_list)
                print('*'*100)
                raise
            else:
                j = t2_index_list.index(str(mark_index))
                mark_index = t2_index_list[j:] 
                break
        else:
            mark_index += len(dialogue_list[i])
    return mark_index

flight_list = [str(x) for x in range(1000, 1030)]

book_book_num = 0
book_no_flight_num = 0
change_change_num = 0
change_no_flight_num = 0
change_no_reservation_num = 0
cancel_cancel_num = 0
cancel_no_reservation_num = 0

error = 0

final_state_dict = {}
total_example = 0

threshold_index = 20

lines = data_fp.readlines()
print('Start generating state label !')
for dialogue_lines in tqdm(lines):

    # dev.kb
    kb_lines = kb_fp.readline()
    resevation = None
    kb_lines = kb_lines.split()
    resevation = kb_lines[0].split('_', 1)[1].split('>', 1)[0]

    # dev.data
    data_items = dialogue_lines.split("|")
    intent_item = data_items[0].split()
    ground_truth = data_items[1].split()
    if len(ground_truth) != 4:
        error_report_fp.write('Error example : ' + str(total_example) + '\n')
        # continue
    dialogue = data_items[2].split()
    t1t2_index = data_items[3].split()
    t1_index_list, t2_index_list = split_index(t1t2_index)
    # print('t1_index_list : ', t1_index_list)
    # print('t2_index_list : ', t2_index_list)
    # break

    # # print to check
    # print('\n\n\n' + '*'*100 )
    # print('total_example : ', total_example)
    # print('intent_item : ', intent_item)
    # print('ground_truth : ', ground_truth)
    # print('dialogue : ', dialogue)
    # print('t1t2_index : ', t1t2_index)
    # print('*'*100 + '\n\n\n')
    
    # split each sentence
    turn = -1
    complete = 0
    sentence_list = []
    sentence  = []
    sentence_index = []
    curr_index = 0
    for word in dialogue:
        complete = 0
        if (turn == 0 and word == '<t2>') or (turn == 1 and word == '<t1>'):
            sentence_list.append(sentence)
            sentence = []
            complete = 1
        if word == '<t1>':
            turn = 0
        if word == '<t2>':
            turn = 1
        sentence.append(word)
        curr_index = curr_index + 1
        # if curr_index == len(dialogue):
        #     sentence_list.append(sentence)
        #     sentence = []
        #     complete = 1

    # intent, ground_truth
    intent = {}
    intent['goal'] = intent_item[14].split('_', 1)[1].split('>', 1)[0]
        
    if len(ground_truth) != 4:
        # print('ground_truth : ', ground_truth, len(ground_truth))
        # raise
        final_state = ground_truth[1].split('_', 1)[1].split('>', 1)[0]
        flight_number_state = ground_truth[0].split('_', 1)[1].split('>', 1)[0]
    else:
        final_state = ground_truth[3].split('_', 1)[1].split('>', 1)[0]
        flight_number_state = ground_truth[2].split('_', 1)[1].split('>', 1)[0]

    if final_state in final_state_dict.keys():
        final_state_dict[final_state] = final_state_dict[final_state] + 1
    else:
        final_state_dict[final_state] = 1 
    st = intent['goal'] + '--' + final_state
    state_combination[st] = state_combination[st] + 1

    turn = -1
    index = 0
    t1_index = -1
    t2_index = -1

    check_flight = 0
    check_no_flight = 0
    check_cancel = 0
    check_reservation = 0
    condiction_type = -1
    total_word = 0
    now_word = 0
    for i in sentence_list:
        total_word += len(i)

    condiction11 = ['I', 'am', 'happy', 'to', 'help', 'you',]

    condiction1 = ['have', 'found', 'got', 'find', 'a', 'an', 'flight', 'ticket', 'airline', 'airlines']
    condiction2 = ['There', 'there', 'here', 'Here', 'is', 'a', 'flight', 'ticket', 'airline', 'airlines']
    condiction3 = ['couple', 'of', 'connection', 'connections', 'connections-', 'connecting', 'flight', 'airline', 'airlines', 'with', 'limit', 'class', 'price', 'Price', 'found', 'find', 'fare', 'fare-', 'airfare', 'book', 'proceed']
    condiction4 = ['your', 'details', 'were', 'matched']

    condiction5 = ['reservation', 'reserved', 'available', 'Reservation', 'booking', 'cancellation', 'cancelled', 'cancel', 'canceled']
    condiction6 = ['found', 'find', 'previous', 'ticket', 'cancelled', 'cancel', 'reservation', 'cancellation', 'booking', 'Reservation']

    condiction7 = ['Sorry', 'sorry', 'no', 'not', "n't", 'No', 'found', 'find', 'seeing', 'exist', 'existed', 'exists', 'Reservation', 'reservation', 'reserved', 'reservations', 'available', 'booking', 'ticket', 'details','name', 'found', 'flight'
                  , 'cancellation', 'cancelled', 'cancel', 'canceled', 'change']
    condiction7_1 = ['know', 'name', 'your']
    condiction8 = ['There', 'there', 'here', 'Here', 'is', 'no', 'not', "n't", 'No', 'flight', 'ticket', 'reservation', 'reserved', 'reservations']

    condiction9 = ['There', 'there', 'here', 'Here', 'is', 'are', 'no', 'not', "n't", 'No', 'flight', 'flights', 'ticket', 'tickets', 'available', 'found']
    condiction10 = ['unable', 'Sorry', 'sorry', 'found', 'find', 'available', 'no', 'not', "n't", 'No', 'plane', 'flight', 'flights',
                     'tickets', 'ticket', 'provide', 'providing']

    condiction12 = ['unable', 'no', 'not', "n't", 'No', 'lights', 'fight', 'fights', 'match', 'matching', 'matches', 'flight', 'flights', 'plane', 'ticket', 'tickets', 'found', 'find', 'available', 'requirement', 'requirements']  
    condiction13 = ['no', 'not', "n't", 'No', 'plane', 'on', 'in', 'as', 'with']   
    condiction14 = ['No', 'no', 'reservation', 'reservations', 'found', 'on']     

    condiction15 = ['no', 'not', "n't", 'No', 'unable', 'Sorry', 'sorry']
    condiction16 = ['lights', 'fight', 'fights', 'match', 'matching', 'matches', 'flight', 'flights', 'route', 'tickets', 'ticket', 'details']          

    for dialogue_item in sentence_list:
        now_word += len(dialogue_item)
        if dialogue_item[0] == '<t1>':
            turn = 0
            t1_index += 1
        if dialogue_item[0] == '<t2>':
            turn = 1
            t2_index += 1

            # book, change
            ############################################################################################################################################################################# 
            if ((index >=5 and final_state == 'book' and (100. * index / len(sentence_list) >= 48.0 )) or (index > 9 and final_state == 'change' and (100. * index / len(sentence_list) >= 48.0 )) or ( index > 7 and intent['goal'] == 'change' and final_state == 'no_flight' and (100. * index / len(sentence_list) >= 48.0 )) or ( index > 5 and intent['goal'] == 'book' and final_state == 'no_flight' and (100. * index / len(sentence_list) >= 48.0 ))) and (any('flight' in s for s in dialogue_item) or any('irline' in s for s in dialogue_item) or any('Flight' in s for s in dialogue_item)):
                # <t2> i found a ...
                check_flight = 1
                check_flight_index = index
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 0
                break

            # # flight number '1000' ~ '1029'
            if index >= 5 and ((final_state == 'book' and (100. * index / len(sentence_list) >= 30.0 )) or (final_state == 'change' and (100. * index / len(sentence_list) >= 46.0 )) or (final_state == 'no_flight' and (100. * index / len(sentence_list) >= 38.0 ))):
                for word in dialogue_item:
                    if word.isdigit() and (int(word) >= 1000 and int(word) <= 1029):
                        check_flight = 1
                        check_flight_index = index
                        check_no_flight = 1
                        check_no_flight_index = index
                        condiction_type = 1
                        break
                if check_flight == 1 or check_no_flight == 1:
                    break

            if ((index > 7 and ((final_state == 'change' and (100. * index / len(sentence_list) >= 46.0 )) or (final_state == 'no_flight' and (100. * index / len(sentence_list) >= 48.0 )))) or (index >=3 and final_state == 'book' and (100. * index / len(sentence_list) >= 48.0 ))) and len(list(set(condiction1).intersection(dialogue_item))) >= 3 and len(list(set(condiction11).intersection(dialogue_item))) <= 5:
                # <t2> 'have', 'found', 'got', 'find', 'a', 'an', 'flight', 'ticket'
                check_flight = 1
                check_flight_index = index
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 2
                break

            if ((index >=3 and final_state == 'book' and (100. * index / len(sentence_list) >= 30.0 )) or (index > 7 and final_state == 'change' and (100. * index / len(sentence_list) >= 46.0 )) or ( index > 3 and intent['goal'] == 'change' and final_state == 'no_flight' and (100. * index / len(sentence_list) >= 48.0 )) or ( index > 3 and intent['goal'] == 'book' and final_state == 'no_flight' and (100. * index / len(sentence_list) >= 48.0 ))) and (len(list(set(condiction2).intersection(dialogue_item))) >= 4 or (any('flight' in s for s in dialogue_item) and len(list(set(condiction2).intersection(dialogue_item))) >= 3)):
                # <t2> 'There', 'there', 'here', 'Here', 'is', 'a', 'flight', 'ticket'
                check_flight = 1
                check_flight_index = index
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 3
                break

            if index >= 3 and ((final_state == 'book' and (100. * index / len(sentence_list) >= 30.0 )) or (final_state == 'change' and (100. * index / len(sentence_list) >= 45.0 )) or (final_state == 'no_flight' and (100. * index / len(sentence_list) >= 45.0 ))) and len(list(set(condiction3).intersection(dialogue_item))) >= 4:
                # <t2> 'connection', 'flight', 'airline', 'with', 'limit', 'price'
                check_flight = 1
                check_flight_index = index
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 4
                break

            if index >= 5 and ((final_state == 'book' and (100. * index / len(sentence_list) >= 30.0 )) or (final_state == 'change' and (100. * index / len(sentence_list) >= 46.0 ))) and len(list(set(condiction4).intersection(dialogue_item))) >= 4:
                # <t2> 'your', 'details', 'were', 'matched'
                check_flight = 1
                check_flight_index = index
                condiction_type = 5
                break

            ############################################################################################################################################################################# 

            if index >= 5 and final_state == 'cancel' and (any('cancel' in s for s in dialogue_item) or any('eserve' in s for s in dialogue_item)):
                # <t2> i found a ...
                check_cancel = 1
                check_cancel_index = index
                condiction_type = 6
                break

            if index >= 5 and (final_state == 'cancel') and len(list(set(condiction5).intersection(dialogue_item))) >= 1:
                # <t2> i found a ...
                check_cancel = 1
                check_cancel_index = index
                condiction_type = 7
                break

            if index >= 3 and (100. * index / len(sentence_list) >= 45.0 )  and (final_state == 'cancel') and len(list(set(condiction6).intersection(dialogue_item))) >= 2:
                # <t2> i found a ...
                check_cancel = 1
                check_cancel_index = index
                condiction_type = 8
                break

            ############################################################################################################################################################################# 

            if index >= 1 and ((100. * index / len(sentence_list) >= 38.0 ) or (100. * now_word / total_word) >= 65.0 ) and  (final_state == 'no_reservation') and len(list(set(condiction7).intersection(dialogue_item))) >= 3 and len(list(set(condiction7_1).intersection(dialogue_item))) != 3:
                # <t2> i found a ...
                check_reservation = 1
                check_reservation_index = index
                condiction_type = 9
                break

            if index >= 1 and ((100. * index / len(sentence_list) >= 38.0 ) or (100. * now_word / total_word) >= 65.0) and (final_state == 'no_reservation') and len(list(set(condiction8).intersection(dialogue_item))) >= 3:
                # <t2> i found a ...
                check_reservation = 1
                check_reservation_index = index
                condiction_type = 10
                break

            if index >= 1 and (final_state == 'no_reservation') and len(list(set(condiction14).intersection(dialogue_item))) >= 3 and (('no' in dialogue_item) or ('No' in dialogue_item)):
                # <t2> i found a ...
                check_reservation = 1
                check_reservation_index = index
                condiction_type = 17
                break

            ############################################################################################################################################################################# 

            if index > 7 and (final_state == 'no_flight') and (100. * index / len(sentence_list) >= 42.0 ) and len(list(set(condiction9).intersection(dialogue_item))) >= 3:
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 11
                break

            if index > 7 and (final_state == 'no_flight') and (100. * index / len(sentence_list) > 45.0 ) and len(list(set(condiction10).intersection(dialogue_item))) >= 2:
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 12
                break

            if index >= 3 and (final_state == 'no_flight') and (100. * index / len(sentence_list) > 45.0 ) and len(list(set(condiction12).intersection(dialogue_item))) >= 3:
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 13
                break

            if index >= 3 and (final_state == 'no_flight') and (100. * index / len(sentence_list) >= 42.0 ) and len(list(set(condiction13).intersection(dialogue_item))) >= 3and ('plane' in dialogue_item):
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 14
                break

            if index >= 3 and (final_state == 'no_flight') and (100. * index / len(sentence_list) >= 42.0 ) and len(list(set(condiction14).intersection(dialogue_item))) >= 3 and ('no' in dialogue_item):
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 15
                break

            if index >= 3 and (final_state == 'no_flight') and (100. * index / len(sentence_list) >= 42.0 ) and len(list(set(condiction15).intersection(dialogue_item))) >= 1 and len(list(set(condiction16).intersection(dialogue_item))) >= 1:
                # <t2> i found a ...
                check_no_flight = 1
                check_no_flight_index = index
                condiction_type = 16
                break

            ############################################################################################################################################################################# 

        index += 1

    if (intent['goal'] == 'book' and final_state == 'book'):
        if check_flight == 1:
            bb_fp.write('Check : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_flight_index, len(sentence_list), 100. * index / len(sentence_list), (100. * now_word / total_word)) + ' type:{:2}'.format(condiction_type))
            bb_fp.write(str(sentence_list[check_flight_index]))
            bb_fp.write('\n')
            book_book_num += 1
            mark_index = check_mark_index(check_flight_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')
        else:
            fail_bb_fp.write('*'*100); fail_bb_fp.write('\n')
            fail_bb_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_bb_fp.write('\n')
            fail_bb_fp.write(str(sentence_list)); fail_bb_fp.write('\n')
            fail_bb_fp.write('Error : ' + str(total_example)); fail_bb_fp.write('\n')
            error += 1
            fail_bb_fp.write ('*'*100); fail_bb_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')

    elif (intent['goal'] == 'change' and final_state == 'change'):
        if check_flight == 1:
            chch_fp.write('Check : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_flight_index, len(sentence_list), 100. * check_flight_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            chch_fp.write(str(sentence_list[check_flight_index]))
            chch_fp.write('\n')
            change_change_num += 1
            mark_index = check_mark_index(check_flight_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')
        else:
            fail_chch_fp.write('*'*100); fail_chch_fp.write('\n')
            fail_chch_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chch_fp.write('\n')
            fail_chch_fp.write(str(sentence_list)); fail_chch_fp.write('\n')
            fail_chch_fp.write('Error : ' + str(total_example)); fail_chch_fp.write('\n')
            error += 1
            fail_chch_fp.write ('*'*100); fail_chch_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')


    elif (intent['goal'] == 'book' and final_state == 'no_flight'):
        if check_no_flight == 1:
            bn_fp.write('Check : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_no_flight_index, len(sentence_list), 100. * check_no_flight_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            bn_fp.write(str(sentence_list[check_no_flight_index]))
            bn_fp.write('\n')
            book_no_flight_num += 1
            mark_index = check_mark_index(check_no_flight_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')
        else:
            fail_bn_fp.write('*'*100); fail_bn_fp.write('\n')
            fail_bn_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_bn_fp.write('\n')
            fail_bn_fp.write(str(sentence_list)); fail_bn_fp.write('\n')
            fail_bn_fp.write('Error : ' + str(total_example)); fail_bn_fp.write('\n')
            error += 1
            fail_bn_fp.write ('*'*100); fail_bn_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')


    elif (intent['goal'] == 'change' and final_state == 'no_flight'):
        if check_no_flight == 1:
            chno_fp.write('Check : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_no_flight_index, len(sentence_list), 100. * check_no_flight_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            chno_fp.write(str(sentence_list[check_no_flight_index]))
            chno_fp.write('\n')
            change_no_flight_num += 1
            mark_index = check_mark_index(check_no_flight_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')
        else:
            fail_chno_fp.write('*'*100); fail_chno_fp.write('\n')
            fail_chno_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chno_fp.write('\n')
            fail_chno_fp.write(str(sentence_list)); fail_chno_fp.write('\n')
            fail_chno_fp.write('Error : ' + str(total_example)); fail_chno_fp.write('\n')
            error += 1
            fail_chno_fp.write ('*'*100); fail_chno_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')

    elif (intent['goal'] == 'cancel' and final_state == 'no_reservation'):
        if check_reservation == 1:
            cnor_fp.write('Cancel--No_reservation : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_reservation_index, len(sentence_list), 100. * check_reservation_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            cnor_fp.write(str(sentence_list[check_reservation_index]))
            cnor_fp.write('\n')
            cancel_no_reservation_num += 1
            mark_index = check_mark_index(check_reservation_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')
        else:
            fail_cnor_fp.write('*'*100); fail_cnor_fp.write('\n')
            fail_cnor_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_cnor_fp.write('\n')
            fail_cnor_fp.write(str(sentence_list)); fail_cnor_fp.write('\n')
            fail_cnor_fp.write('Error : ' + str(total_example)); fail_cnor_fp.write('\n')
            error += 1
            fail_cnor_fp.write ('*'*100); fail_cnor_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')

    elif (intent['goal'] == 'change' and final_state == 'no_reservation'):
        if check_reservation == 1:
            chnor_fp.write('Change--No_reservation : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_reservation_index, len(sentence_list), 100. * check_reservation_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            chnor_fp.write(str(sentence_list[check_reservation_index]))
            chnor_fp.write('\n')
            change_no_reservation_num += 1
            mark_index = check_mark_index(check_reservation_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')
        else:
            fail_chnor_fp.write('*'*100); fail_chnor_fp.write('\n')
            fail_chnor_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chnor_fp.write('\n')
            fail_chnor_fp.write(str(sentence_list)); fail_chnor_fp.write('\n')
            fail_chnor_fp.write('Error : ' + str(total_example)); fail_chnor_fp.write('\n')
            error += 1
            fail_chnor_fp.write ('*'*100); fail_chnor_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')

    elif (intent['goal'] == 'cancel' and final_state == 'cancel'):
        if check_cancel == 1:
            cc_fp.write('Cancel--Cancel : {:<4}/{:<4} {:>4.1f}|{:>4.1f}'.format(check_cancel_index, len(sentence_list), 100. * check_cancel_index / len(sentence_list), (100. * now_word / total_word))+ ' type:{:2}'.format(condiction_type))
            cc_fp.write(str(sentence_list[check_cancel_index]))
            cc_fp.write('\n')
            cancel_cancel_num += 1
            mark_index = check_mark_index(check_cancel_index, sentence_list, t2_index_list)
            state_tracking_fp.write('{:<5}'.format(str(mark_index).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')
        else:
            fail_cc_fp.write('*'*100); fail_cc_fp.write('\n')
            fail_cc_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_cc_fp.write('\n')
            fail_cc_fp.write(str(sentence_list)); fail_cc_fp.write('\n')
            fail_cc_fp.write('Error : ' + str(total_example)); fail_cc_fp.write('\n')
            error += 1
            fail_cc_fp.write ('*'*100); fail_cc_fp.write('\n')
            # state_tracking_fp.write(' ? ' + ' | ' + str(t2_index_list).strip('[]') + '\n')
            s = list(t2_index_list[-2])
            state_tracking_fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')
            
        # print(dialogue_item)
    else:
        print(dialogue_item)
        raise

    total_example += 1


#########################################################################################################################################
#########################################################################################################################################

data_fp.close()
kb_fp.close()
state_tracking_fp.close()

log_fp.write('*'*50 +'\n')
log_fp.write('total_example : ' +  str(total_example) + ' \n')
for key, value in state_combination.items() :
    log_fp.write('    >>  {:26} : {:<6} prob : {:>4.1f} % \n'.format(key,  value, 100. * value /sum(state_combination.values())))
log_fp.write('Book_Book : ' + str(book_book_num) + ' / ' + str(state_combination['book--book']) +  ' : ' +  str(state_combination['book--book']-book_book_num) + ' \n')
log_fp.write('Book_No_flight : ' + str(book_no_flight_num) + ' / ' + str(state_combination['book--no_flight']) + ' : ' + str(state_combination['book--no_flight']-book_no_flight_num) + ' \n')
log_fp.write('Change_Change : ' + str(change_change_num) + ' / ' + str(state_combination['change--change']) + ' : ' + str(state_combination['change--change'] - change_change_num) + ' \n')
log_fp.write('Change_No_flight : ' + str(change_no_flight_num) + ' / ' + str(state_combination['change--no_flight']) + ' : ' + str(state_combination['change--no_flight']-change_no_flight_num) + ' \n')
log_fp.write('Change_No_reservation : ' + str(change_no_reservation_num) + ' / ' + str(state_combination['change--no_reservation']) + ' : ' + str(state_combination['change--no_reservation'] - change_no_reservation_num) + ' \n')
log_fp.write('Cancel_Cancel : ' + str(cancel_cancel_num) +' / ' + str(state_combination['cancel--cancel']) + ' : ' + str(state_combination['cancel--cancel'] - cancel_cancel_num) + ' \n')
log_fp.write('Cancel_No_reservation : ' + str(cancel_no_reservation_num) + ' / ' + str(state_combination['cancel--no_reservation']) + ' : ' + str(state_combination['cancel--no_reservation'] - cancel_no_reservation_num) + ' \n')
log_fp.write('Error : ' + str(error) + ' \n')

if total_example != sum(state_combination.values()):
    log_fp.write('Number Sum Error ! \n')
else:
    log_fp.write('Number OK ! \n')
log_fp.write('*'*50 +'\n')
print('Finish generating state label !')
# SQL line, read dialogue line, read kb line
# sql_fp = open('train.sql.txt', 'w')
# data_fp = open('/home/sclab/pytorch-seq2seq/data/sub_air/train_sub.data', "r")
# kb_fp = open('/home/sclab/pytorch-seq2seq/data/sub_air/train_sub.kb', "r")
data_fp = open('/home/sclab/airdialogue_model-master/data/airdialogue/tokenized/train.data', "r")
kb_fp = open('/home/sclab/airdialogue_model-master/data/airdialogue/tokenized/train.kb', "r")
check_num = 5000

bb_fp = open('label_result/Book_Book.txt', 'w')
bn_fp = open('label_result/Book_No_flight.txt', 'w')
chch_fp = open('label_result/Change_Change.txt', 'w')
chno_fp = open('label_result/Change_No_flight.txt', 'w')
chnor_fp = open('label_result/Change_No_reservation.txt', 'w')
cc_fp = open('label_result/Cancel_Cancel.txt', 'w')
cnor_fp = open('label_result/Cancel_No_reservation.txt', 'w')

fail_bb_fp = open('label_result_fail/Book_Book_fail.txt', 'w')
fail_bn_fp = open('label_result_fail/Book_No_flight_fail.txt', 'w')
fail_chch_fp = open('label_result_fail/Change_Change_fail.txt', 'w')
fail_chno_fp = open('label_result_fail/Change_No_flight_fail.txt', 'w')
fail_chnor_fp = open('label_result_fail/Change_No_reservation_fail.txt', 'w')
fail_cc_fp = open('label_result_fail/Cancel_Cancel_fail.txt', 'w')
fail_cnor_fp = open('label_result_fail/Cancel_No_reservation_fail.txt', 'w')

# intent :: train.json
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

for dialogue_lines in iter(data_fp):

    kb_lines = kb_fp.readline()
    intent = {}
    resevation = None
    sql = 'SELECT * FROM Airdialogue_Table WHERE '
    index = 0
    dialogue_lines = dialogue_lines.split() # space + \n
    kb_lines = kb_lines.split()

    flight_number_state = None
    final_state = None
    check_flight = 0
    check_no_flight = 0
    check_cancel = 0
    check_cancel = 0
    check_reservation = 0
    turn = -1

    for dialogue_item in dialogue_lines:

        if dialogue_item == '<t1>':
            turn = 0

        if dialogue_item == '<t2>':
            turn = 1

        if index == 0: # has_reservation_probability = 0.10
            resevation = kb_lines[0].split('_', 1)[1].split('>', 1)[0]

        elif index == 14: # ['book', 'change', 'cancel'] [0.80, 0.1, 0.1]
            intent['goal'] = dialogue_item.split('_', 1)[1].split('>', 1)[0]

        elif '<fl_' in dialogue_item : # <fl_empty> <fl_1000> ~ <fl_1029>
            flight_number_state = dialogue_item.split('_', 1)[1].split('>', 1)[0]

        elif '<st_' in dialogue_item : # <st_no_reservation> <st_no_flight> <st_book> <st_cancel> <st_change>
            final_state = dialogue_item.split('_', 1)[1].split('>', 1)[0]
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in final_state_dict.keys():
                final_state_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = final_state_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                final_state_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif dialogue_item.isdigit() and (int(dialogue_item) >= 1000 and int(dialogue_item) <= 1029) and turn == 1 and index > threshold_index: # <t2>, '1000' ~ '1029', dialogue
            check_flight = 1
            check_flight_index = index

        elif 'airline' in dialogue_item and (final_state == 'book' or final_state == 'change') and turn == 1 and index > threshold_index: 
            # <t2> xxx airline
            check_flight = 1
            check_flight_index = index

        elif 'found' in dialogue_item and (final_state == 'book' or final_state == 'change') and turn == 1 and index > threshold_index: 
            # <t2> i found a ...
            check_flight = 1
            check_flight_index = index

        elif 'cancel' in dialogue_item and final_state == 'cancel' and turn == 1 and index > threshold_index: 
            # <t2> cancel ...
            check_cancel = 1
            check_cancel_index = index

        elif (('ere' in dialogue_item) and ('is' == dialogue_lines[index+1]) and ('a' == dialogue_lines[index+2])) and turn == 1 and index > threshold_index:
            # <t2> there is a (flight, ticket ... )
            check_flight = 1
            check_flight_index = index
            check_no_flight = 1
            check_no_flight_index = index

        elif (('flight' in dialogue_item) and ('a' == dialogue_lines[index-1])) and turn == 1 and index > threshold_index:
            # <t2> ... a flight ...
            check_flight = 1
            check_flight_index = index
            check_no_flight = 1
            check_no_flight_index = index

        elif (('reservation' in dialogue_item) or ('reserv' in dialogue_item)) and turn == 1 and index > threshold_index: 
            # <t2> ... reservation (reserved) ...
            check_reservation = 1
            check_reservation_index = index

        elif (('reserv' in dialogue_item) or ('available' in dialogue_item) or ('booking' in dialogue_item) or ('eserva' in dialogue_item) or ('reseravtion' in dialogue_item) ) and final_state == 'no_reservation' and turn == 1 and index > threshold_index:
            # <t2> ... reservation (reserved), available, booking ...
            check_reservation = 1
            check_reservation_index = index

        elif ((('no' in dialogue_item) or ('No' in dialogue_item)) and (('ticket' in dialogue_lines[index+1]) or ('name' in dialogue_lines[index+1]) or ('find' in dialogue_lines[index+1]) or ('found' in dialogue_lines[index+1]) or ('flight' in dialogue_lines[index+1]))) and final_state == 'no_reservation' and turn == 1 and index > threshold_index:
            # <t2> ... no (ticket, name, found, find, flight) ...
            check_reservation = 1
            check_reservation_index = index

        elif ((('no' in dialogue_item) or ('No' in dialogue_item)) and (('flight' in dialogue_lines[index+1]) or ('flight' in dialogue_lines[index+2]) or ('flight' in dialogue_lines[index+3]))) and final_state == 'no_flight' and turn == 1 and index > threshold_index:
            # <t2> ... no (flight) ...
            check_no_flight = 1
            check_no_flight_index = index

        elif ((('ot' in dialogue_item) or ("'t" in dialogue_item)) and (('found' in dialogue_lines[index+1]) or ('find' in dialogue_lines[index+1]))) and final_state == 'no_flight' and turn == 1 and index > threshold_index:
            # <t2> ... not (found) ...
            check_no_flight = 1
            check_no_flight_index = index

        elif ('orry' in dialogue_item) and final_state == 'no_flight' and turn == 1 and index > threshold_index: 
            # <t2> i found a ...
            check_no_flight = 1
            check_no_flight_index = index

        elif ('orry' in dialogue_item) and final_state == 'no_reservation' and turn == 1 and index > threshold_index: 
            # <t2> i found a ...
            check_reservation = 1
            check_reservation_index = index

        elif (('found' in dialogue_item) or ('find' in dialogue_item)) and final_state == 'no_flight' and turn == 1 and index > threshold_index: 
            # <t2> i found a ...
            check_no_flight = 1
            check_no_flight_index = index

        elif ('unable' in dialogue_item) and final_state == 'no_flight' and turn == 1 and index > threshold_index: 
            # <t2> i found a ...
            check_no_flight = 1
            check_no_flight_index = index

        elif ('flight' in dialogue_item) and final_state == 'no_flight' and turn == 1 and (100. * index/len(dialogue_lines)) > 75.: 
            # <t2> i found a ...
            check_no_flight = 1
            check_no_flight_index = index

        for f_number in flight_list: # '1000' ~ '1029' in string, <t2> , dialogue 
            # print(f_number)
            if (f_number in dialogue_item) and turn == 1 and index > threshold_index:
                # print('Find : ', f_number, ' <--> ', dialogue_item)
                check_flight = 1
                check_flight_index = index
                check_no_flight = 1
                check_no_flight_index = index

        index += 1
        
    if (intent['goal'] == 'cancel' and final_state == 'cancel'):
        if check_reservation == 1:
            cc_fp.write('Cancel--Cancel : {:<4}/{:<4} {:>4.1f}'.format(check_reservation_index, len(dialogue_lines), 100. * check_reservation_index / len(dialogue_lines)))
            cc_fp.write(str(dialogue_lines[check_reservation_index-7:check_reservation_index+7]))
            cc_fp.write('\n')
            cancel_cancel_num += 1
        elif check_cancel == 1:
            cc_fp.write('Cancel--Cancel : {:<4}/{:<4} {:>4.1f}'.format(check_cancel_index, len(dialogue_lines), 100. * check_cancel_index / len(dialogue_lines)))
            cc_fp.write(str(dialogue_lines[check_cancel_index-7:check_cancel_index+7]))
            cc_fp.write('\n')
            cancel_cancel_num += 1
        else:
            fail_cc_fp.write('*'*100); fail_cc_fp.write('\n')
            fail_cc_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_cc_fp.write('\n')
            fail_cc_fp.write(str(dialogue_lines)); fail_cc_fp.write('\n')
            fail_cc_fp.write('Error : ' + str(total_example)); fail_cc_fp.write('\n')
            error += 1
            fail_cc_fp.write ('*'*100); fail_cc_fp.write('\n')

    if (intent['goal'] == 'cancel' and final_state == 'no_reservation'):
        if check_reservation == 1:
            cnor_fp.write('Cancel--No_reservation : {:<4}/{:<4} {:>4.1f}'.format(check_reservation_index, len(dialogue_lines), 100. * check_reservation_index / len(dialogue_lines)))
            cnor_fp.write(str(dialogue_lines[check_reservation_index-7:check_reservation_index+7]))
            cnor_fp.write('\n')
            cancel_no_reservation_num += 1
        else:
            fail_cnor_fp.write('*'*100); fail_cnor_fp.write('\n')
            fail_cnor_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_cnor_fp.write('\n')
            fail_cnor_fp.write(str(dialogue_lines)); fail_cnor_fp.write('\n')
            fail_cnor_fp.write('Error : ' + str(total_example)); fail_cnor_fp.write('\n')
            error += 1
            fail_cnor_fp.write ('*'*100); fail_cnor_fp.write('\n')

    if (intent['goal'] == 'change' and final_state == 'no_reservation'):
        if check_reservation == 1:
            chnor_fp.write('Change--No_reservation : {:<4}/{:<4} {:>4.1f}'.format(check_reservation_index, len(dialogue_lines), 100. * check_reservation_index / len(dialogue_lines)))
            chnor_fp.write(str(dialogue_lines[check_reservation_index-7:check_reservation_index+7]))
            chnor_fp.write('\n')
            change_no_reservation_num += 1
        else:
            fail_chnor_fp.write('*'*100); fail_chnor_fp.write('\n')
            fail_chnor_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chnor_fp.write('\n')
            fail_chnor_fp.write(str(dialogue_lines)); fail_chnor_fp.write('\n')
            fail_chnor_fp.write('Error : ' + str(total_example)); fail_chnor_fp.write('\n')
            error += 1
            fail_chnor_fp.write ('*'*100); fail_chnor_fp.write('\n')

    if (intent['goal'] == 'book' and final_state == 'book'):
        if check_flight == 1:
            bb_fp.write('Check : {:<4}/{:<4} {:>4.1f}'.format(check_flight_index, len(dialogue_lines), 100. * check_flight_index / len(dialogue_lines)))
            bb_fp.write(str(dialogue_lines[check_flight_index-7:check_flight_index+7]))
            bb_fp.write('\n')
            book_book_num += 1
        else:
            fail_bb_fp.write('*'*100); fail_bb_fp.write('\n')
            fail_bb_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_bb_fp.write('\n')
            fail_bb_fp.write(str(dialogue_lines)); fail_bb_fp.write('\n')
            fail_bb_fp.write('Error : ' + str(total_example)); fail_bb_fp.write('\n')
            error += 1
            fail_bb_fp.write ('*'*100); fail_bb_fp.write('\n')

    if (intent['goal'] == 'change' and final_state == 'change'):
        if check_flight == 1:
            chch_fp.write('Check : {:<4}/{:<4} {:>4.1f}'.format(check_flight_index, len(dialogue_lines), 100. * check_flight_index / len(dialogue_lines)))
            chch_fp.write(str(dialogue_lines[check_flight_index-7:check_flight_index+7]))
            chch_fp.write('\n')
            change_change_num += 1
        else:
            fail_chch_fp.write('*'*100); fail_chch_fp.write('\n')
            fail_chch_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chch_fp.write('\n')
            fail_chch_fp.write(str(dialogue_lines)); fail_chch_fp.write('\n')
            fail_chch_fp.write('Error : ' + str(total_example)); fail_chch_fp.write('\n')
            error += 1
            fail_chch_fp.write ('*'*100); fail_chch_fp.write('\n')

    if (intent['goal'] == 'book' and final_state == 'no_flight'):
        if check_no_flight == 1:
            bn_fp.write('Check : {:<4}/{:<4} {:>4.1f}'.format(check_no_flight_index, len(dialogue_lines), 100. * check_no_flight_index / len(dialogue_lines)))
            bn_fp.write(str(dialogue_lines[check_no_flight_index-7:check_no_flight_index+7]))
            bn_fp.write('\n')
            book_no_flight_num += 1
        else:
            fail_bn_fp.write('*'*100); fail_bn_fp.write('\n')
            fail_bn_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_bn_fp.write('\n')
            fail_bn_fp.write(str(dialogue_lines)); fail_bn_fp.write('\n')
            fail_bn_fp.write('Error : ' + str(total_example)); fail_bn_fp.write('\n')
            error += 1
            fail_bn_fp.write ('*'*100); fail_bn_fp.write('\n')

    if (intent['goal'] == 'change' and final_state == 'no_flight'):
        if check_no_flight == 1:
            chno_fp.write('Check : {:<4}/{:<4} {:>4.1f}'.format(check_no_flight_index, len(dialogue_lines), 100. * check_no_flight_index / len(dialogue_lines)))
            chno_fp.write(str(dialogue_lines[check_no_flight_index-7:check_no_flight_index+7]))
            chno_fp.write('\n')
            change_no_flight_num += 1
        else:
            fail_chno_fp.write('*'*100); fail_chno_fp.write('\n')
            fail_chno_fp.write('Goal : {} Final_state : {}'.format(intent['goal'], final_state)); fail_chno_fp.write('\n')
            fail_chno_fp.write(str(dialogue_lines)); fail_chno_fp.write('\n')
            fail_chno_fp.write('Error : ' + str(total_example)); fail_chno_fp.write('\n')
            error += 1
            fail_chno_fp.write ('*'*100); fail_chno_fp.write('\n')
    
    # print('Goal : ', intent['goal'])
    # print('resevation : ', resevation)
    # print('flight_number_state : ', flight_number_state)
    # print('final_state : ', final_state)

    st = intent['goal'] + '--' + final_state
    state_combination[st] = state_combination[st] + 1
    total_example += 1
    # if total_example > check_num:
    #     break

data_fp.close()
kb_fp.close()

print('*'*50)
print('total_example : ', total_example)
for key, value in state_combination.items() :
    print ('    >>  {:26} : {:<6} prob : {:>4.1f} %'.format(key,  value, 100. * value /sum(state_combination.values())))
print('Book_Book : ', book_book_num, ' / ', state_combination['book--book'], ' : ', state_combination['book--book']-book_book_num)
print('Book_No_flight : ', book_no_flight_num, ' / ', state_combination['book--no_flight'], ' : ', state_combination['book--no_flight']-book_no_flight_num)
print('Change_Change : ', change_change_num, ' / ', state_combination['change--change'], ' : ', state_combination['change--change'] - change_change_num)
print('Change_No_flight : ', change_no_flight_num, ' / ', state_combination['change--no_flight'], ' : ', state_combination['change--no_flight']-change_no_flight_num)
print('Change_No_reservation : ', change_no_reservation_num, ' / ', state_combination['change--no_reservation'], ' : ', state_combination['change--no_reservation'] - change_no_reservation_num)
print('Cancel_Cancel : ', cancel_cancel_num, ' / ', state_combination['cancel--cancel'], ' : ', state_combination['cancel--cancel'] - cancel_cancel_num)
print('Cancel_No_reservation : ', cancel_no_reservation_num, ' / ', state_combination['cancel--no_reservation'], ' : ', state_combination['cancel--no_reservation'] - cancel_no_reservation_num)
print('Error : ', error)

if total_example != sum(state_combination.values()):
    print('Number Sum Error !')
else:
    print('Number OK !')
print('*'*50)

# # read kb line
# kb_fp = open('/home/sclab/pytorch-seq2seq/data/sub_air/train_sub.kb', "r")
# for kb_lines in iter(kb_fp):
#     print(kb_lines)
#     # for kb_item in kb_lines:
#     break

# kb_fp.close()


# 4267(not complete), 15410(change, no_flight), 25401(not complete), 45104(change, no_flight), 4392(book, no_flight), 296(no name), 281946(cancel, no_reservation)
# 2503 (cancel, no_flight) 8231(cancel, no_flight) 48547(change, no_flight), 33128(cancel, cancel), 55571(cancel, cancel), 38342(cancel, cancel), 216513(book, no_reservation)
# 210957(not complete) 3004(book, no_reservation) 98259(not complete) 43106(not complete), 15406(change, no_flight), 48547(change, no_flight), 69107(change, no_flight)
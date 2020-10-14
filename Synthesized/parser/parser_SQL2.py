import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--self_play_eval', action='store_true')
args = parser.parse_args()

# SQL line, read dialogue line, read kb line
if args.dev:
    sql_fp = open('../data/synthesized/SQL/dev/dev.sql.txt', 'w')
    data_fp = open('../data/synthesized/tokenized/dev.eval.data', 'r')
    kb_fp = open('../data/synthesized/tokenized/dev.eval.kb', 'r')
elif args.train:
    sql_fp = open('../data/synthesized/SQL/train.sql.txt', 'w')
    data_fp = open('../data/synthesized/tokenized/train.data', "r")
    kb_fp = open('../data/synthesized/tokenized/train.kb', "r")
elif args.self_play_eval:
    sql_fp = open('../data/synthesized/SQL/dev_self_play_eval/self_play_eval.sql.txt', 'w')
    data_fp = open('../data/synthesized/tokenized/dev.selfplay.eval.data', "r")
    kb_fp = open('../data/synthesized/tokenized/dev.selfplay.eval.kb', "r")
else:
    print('Pleae use --dev or --train or --self_play_eval !')
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
airport_list = ['DEN', 'LAX', 'MSP', 'DFW', 'SEA', 'ATL', 'IAH', 'DTW', 'ORD', 'IAD', 'CLT', 'EWR', 'LGA', 'JFK', 'HOU', 'SFO', 'AUS', 'OAK', 'LAS', 'PHL', 'BOS', 'MCO', 'DCA', 'PHX']
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
time_list = ['morning', 'afternoon', 'evening']
air_class_list = ['economy', 'business']
max_price_list = ['200', '500', '1000', '5000']
airline_preference_list = ['normal-cost']

total_example = 0
for dialogue_lines in iter(data_fp):
    kb_lines = kb_fp.readline()
    intent = {}
    resevation = None
    sql = 'SELECT * FROM Airdialogue_Table WHERE '
    index = 0
    dialogue_lines = dialogue_lines.split() # space + \n

    for dialogue_item in dialogue_lines:
        if index == 0: # has_reservation_probability = 0.10
            resevation = kb_lines[0]

        if index == 0: # <a1_IAD> uniform 24
            intent['departure_airport'] = dialogue_item 

            if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in airport_list :
                label = airport_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
            else:
                print('airport_list : ', airport_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                raise
            sql = sql + "departure_airport = '" + str(label) + "'"

            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in departure_airport_dict.keys():
                departure_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = departure_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                departure_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 1: # <a2_ATL> uniform 24
            intent['return_airport'] = dialogue_item

            if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in airport_list :
                label = airport_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
            else:
                print('airport_list : ', airport_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                raise

            sql = sql + " AND return_airport = '" + str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in return_airport_dict.keys():
                return_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = return_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                return_airport_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 2: # <m1_Sept> uniform
            intent['departure_month'] = dialogue_item

            if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in month_list :
                label = month_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
            else:
                print('month_list : ', month_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                raise

            sql = sql + " AND departure_month = '" + str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in departure_month_dict.keys():
                departure_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = departure_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                departure_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 3: # <m2_Sept> uniform
            intent['return_month'] = dialogue_item

            if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in month_list :
                label = month_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
            else:
                print('month_list : ', month_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                raise

            sql = sql + " AND return_month = '" + str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in return_airport_dict.keys():
                return_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = return_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                return_month_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 4: # <d1_11> uniform
            intent['departure_day'] = dialogue_item
            sql = sql + " AND departure_day = " + str(int(dialogue_item.split('_', 1)[1].split('>', 1)[0])-1)
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in departure_day_dict.keys():
                departure_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = departure_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                departure_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 5: # <d2_13> uniform
            intent['return_day'] = dialogue_item
            sql = sql + " AND return_day = " + str(int(dialogue_item.split('_', 1)[1].split('>', 1)[0])-1)
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in return_day_dict.keys():
                return_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = return_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                return_day_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 6: # <t1_all> <t1_evening> <t1_afternoon> <t1_morning> ['morning', 'afternoon', 'evening', 'all'] [0.03, 0.04, 0.03, 0.9]
            intent['departure_time'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] != 'all':

                if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in time_list :
                    label = time_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
                else:
                    print('time_list : ', time_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                    raise

                sql = sql + " AND departure_time_num = '" + str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in departure_time_dict.keys():
                departure_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = departure_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                departure_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 7: # <t1_all> <t1_evening> <t1_afternoon> <t1_morning> ['morning', 'afternoon', 'evening', 'all'] [0.03, 0.04, 0.03, 0.9]
            intent['return_time'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] != 'all':
                
                if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in time_list :
                    label = time_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
                else:
                    print('time_list : ', time_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                    raise

                sql = sql + " AND return_time_num = '" + str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in return_time_dict.keys():
                return_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = return_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                return_time_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 8:
            intent['name1'] = dialogue_item
            if dialogue_item in name1_dict.keys():
                name1_dict[dialogue_item] = name1_dict[dialogue_item] + 1
            else:
                name1_dict[dialogue_item] = 1

        elif index == 9:
            intent['name2'] = dialogue_item
            if dialogue_item in name2_dict.keys():
                name2_dict[dialogue_item] = name2_dict[dialogue_item] + 1
            else:
                name2_dict[dialogue_item] = 1

        elif index == 10: # <cl_all> <cl_business> <cl_economy> ['all', 'economy', 'business'] [0.9, 0.07, 0.03]
            intent['class'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] != 'all':
                
                if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in air_class_list :
                    label = air_class_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
                    if label == 2:
                        raise
                else:
                    print('air_class_list : ', air_class_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                    raise

                sql = sql + " AND class = '"+ str(label) + "'"
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in class_dict.keys():
                class_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = class_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                class_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1
                    
        elif index == 11: # <pr_200> <pr_500> <pr_1000> <pr_5000> [200, 500, 1000, 5000]
            intent['max_price'] = dialogue_item

            if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in max_price_list :
                label = max_price_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
            else:
                print('max_price_list : ', max_price_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                raise

            sql = sql + " AND price <= "+ str(label)
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in max_price_dict.keys():
                max_price_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = max_price_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                max_price_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 12: # <cn_0> <cn_1> <cn_2> [0, 1, any] [0.07, 0.9, 0.03]
            intent['max_connections'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] != '2':
                sql = sql + " AND num_connections <= "+ dialogue_item.split('_', 1)[1].split('>', 1)[0]
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in max_connections_dict.keys():
                max_connections_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = max_connections_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                max_connections_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 13: # ['normal-cost', 'all'] [0.05, 0.95]
            intent['airline_preference'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] != 'all':

                if  dialogue_item.split('_', 1)[1].split('>', 1)[0] in airline_preference_list :
                    label = airline_preference_list.index(dialogue_item.split('_', 1)[1].split('>', 1)[0])
                else:
                    print('airline_preference_list : ', airline_preference_list, ' search : ', dialogue_item.split('_', 1)[1].split('>', 1)[0])
                    raise

                # sql = sql + " AND (airline_preference = 'AA' OR airline_preference = 'Delta' OR airline_preference = 'UA' OR airline_preference = 'Hawaiian')"
                sql = sql + " AND airline_preference = " + str(label) 
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in airline_preference_dict.keys():
                airline_preference_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = airline_preference_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                airline_preference_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        elif index == 14: # ['book', 'change', 'cancel'] [0.80, 0.1, 0.1]
            intent['goal'] = dialogue_item
            if dialogue_item.split('_', 1)[1].split('>', 1)[0] in goal_dict.keys():
                goal_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = goal_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] + 1
            else:
                goal_dict[dialogue_item.split('_', 1)[1].split('>', 1)[0]] = 1

        index += 1
    total_example += 1
    sql_fp.write(sql+'\n')
    if total_example < 10:
        print(sql, '\n')

data_fp.close()
kb_fp.close()
sql_fp.close()

# print check all name
i=0
for mydic in intent_list:
    print('*'*50)
    print('{:26} : {}'.format(intent_name_list[i], len(mydic.keys())))
    i +=1
    for key, value in mydic.items() :
        print('    >>  {:26} : {:<6} prob : {:>4.1f} %'.format(key,  value, 100. * value /sum(mydic.values())))
print('*'*50)

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
args = parser.parse_args()
label_path = './synthesized_label/'

if args.train:

    data_fp = open('../data/synthesized/tokenized/train/train.data', 'r')
    fp = open('../data/synthesized/SQL/State_Tracking.txt', 'w')
    log_fp = open(label_path + 'train.status.log', 'w')

elif args.dev:

    data_fp = open('../data/synthesized/tokenized/dev/dev.eval.data', 'r')
    fp = open('../data/synthesized/SQL/dev/State_Tracking.txt', 'w')
    log_fp = open(label_path + 'dev.status.log', 'w')

else:

    print('Pleae use --dev or --train!')
    raise

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

for lines in iter(data_fp):
    data_items = lines.split("|")
    # intent_item = data_items[0].split()
    ground_truth = data_items[1].split()
    if len(ground_truth) != 4:
        log_fp.write('Error example : ' + str(total_example) + '\n')
        # continue
    dialogue = data_items[2].split()
    t1t2_index = data_items[3].split()
    t1_index_list, t2_index_list = split_index(t1t2_index)

    y_sql = ''
    n_sql = ''
    idx = 0
    if 'I can not find any flight' in data_items[2] or 'We have flight' in data_items[2]:
        have_fli = data_items[2].find('We have flight')
        not_find = data_items[2].find('I can not find any flight')
        if have_fli == -1:
        	y_sql = 'not find'
        	idx = not_find
        else:
        	y_sql = 'have flight'
        	idx = have_fli
    elif 'I am able to locate' in data_items[2]:
        # cancel = data_items[2].find('I am able to locate')
        n_sql = 'cancel'
        idx = data_items[2].find('I am able to locate')
    elif 'I can not locate' in data_items[2]:
        # not_can = data_items[2].find('I can not locate')
        n_sql = 'not cancel or change'
        idx = data_items[2].find('I can not locate')
    else:
        print(data_items[2])

    # print(t2_index_list)
    char_len = 0
    word_len = 0
    for word in dialogue:
    	char_len += len(word)+1
    	word_len += 1
    	if char_len == idx:
    		break

    s = []
    for i in range(len(t2_index_list)):
    	if i == len(t2_index_list)-1:
    		s = list(t2_index_list[i:])
    		break
    	if int(t2_index_list[i]) < word_len and word_len < int(t2_index_list[i+1]):
    		s = list(t2_index_list[i:])
    		break

    if y_sql=='not find' or y_sql=='have flight':
    	fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | Y_SQL ' + '\n')
    else:
    	fp.write('{:<5}'.format(str(s).strip('[]')) + ' | ' + str(t2_index_list).strip('[]') + ' | N_SQL ' + '\n')

    # break
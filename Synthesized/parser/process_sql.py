import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--train', action='store_true', help='Use train')
parser.add_argument('--self_play_eval', action='store_true')
args = parser.parse_args()

if args.dev:
    readf = open('../data/synthesized/SQL/dev/dev.sql.txt', 'r')
    writef = open('../data/synthesized/SQL/dev/dev_tok.jsonl', 'w')
elif args.train:
    readf = open('../data/synthesized/SQL/train.sql.txt', 'r')
    writef = open('../data/synthesized/SQL/train_tok.jsonl', 'w')
elif args.self_play_eval:
    readf = open('../data/synthesized/SQL/dev_self_play_eval/self_play_eval.sql.txt', 'r')
    writef= open('../data/synthesized/SQL/dev_self_play_eval/self_play_eval_tok.jsonl', 'w')
else:
    print('Pleae use --dev or --train or --self_play_eval!')
    raise

lines = readf.readlines()

def col2idx(cond_col):
    if cond_col == 'departure_airport':
        return 0
    elif cond_col == 'return_airport':
        return 1
    elif cond_col == 'departure_month':
        return 2
    elif cond_col == 'return_month':
        return 3
    elif cond_col == 'departure_day':
        return 4
    elif cond_col == 'return_day':
        return 5
    elif cond_col == 'departure_time_num':
        return 6
    elif cond_col == 'return_time_num':
        return 7
    elif cond_col == 'class':
        return 8
    elif cond_col == 'price':
        return 9
    elif cond_col == 'num_connections':
        return 10
    elif cond_col == 'airline_preference':
        return 11
    else:
        print(cond_col, "not in columns!")
        raise

for line in lines:
    sample = dict()
    # query #
    # if line.find('airline') > -1:
    #     # exclude airline preference from SQL query
    #     sample['query'] = line[0:line.find('airline')-6].replace('\'', \
    #         '').replace('<=', 'LTEQL').replace('=', 'EQL')
    # else:

    # print('line : ', line)
    # print('line find : ', line.index('IAD'))
    # print('line : ', line[0:-1].replace('\'', ''))
    # print('beg : ', line.find('WHERE')+6)
    
    # sample['query'] = line[0:-1].replace('\'', '').replace('<=', 'LTEQL').replace('=', 'EQL')
    sample['query'] = line[0:-1].replace('\'', '')
    # conditions #
    sql = dict()
    conds = []
    beg, end = line.find('WHERE')+6, line.find('AND') # each character  beg = 38
    while beg < len(line):
        cond = []
        cur_str = line[beg:end]
        # # exclude airline preference
        # if cur_str.find('airline') > -1:
        #     break
        # col & op #
        if cur_str.find('<=') > -1:
            # col #
            cond_col = cur_str[0 : cur_str.find('<=')-1]
            cond.append(col2idx(cond_col))
            # op #
            cond.append(1) # <=
        else:
            # col #
            cond_col = cur_str[0 : cur_str.find('=')-1]
            cond.append(col2idx(cond_col))
            # op #
            cond.append(0) # =
        # str #
        cond.append(int(cur_str[cur_str.find('=')+2:-1].replace('\'', '')))
        # print('cur_str', cur_str[cur_str.find('=')+2:-1].replace('\'', '') )
        # # # #
        conds.append(cond)
        beg = end+4
        end = len(line) if line.find('AND', end+1) == -1 else line.find('AND', end+1)

    sql['conds'] = conds
    sample['sql'] = sql
    sample['table_id'] = '0-0000' 

    json.dump(sample, writef)
    writef.write('\n')

print('Process Done ! ')


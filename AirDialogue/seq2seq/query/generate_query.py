from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch.autograd import Variable
import math
import unicodedata

loss_action =  nn.CrossEntropyLoss()
loss =  nn.CrossEntropyLoss(reduction='none')
CE = nn.CrossEntropyLoss(reduction='none')
log_softmax = nn.LogSoftmax()
bce_logit = nn.BCEWithLogitsLoss()
sigm = nn.Sigmoid()

def search_t2(gate):
    curr = 0
    for i in range(gate.size(0)):
        if gate[i] == 1:
            curr = i
            break
    return curr

def total_match(A, Q):
    Q = Q[38:].split('AND')
    A = A[38:].split('AND')
    total = 0
    correct = 0
    for i in range(len(Q)):
        if Q[i] in A:
            correct += 1
    total = len(Q)

    # print('output_query len : ', len(Q), Q)
    # print('truth_query len : ', len(A), A)
    # print('correct : ', correct, ' total : ', total)
    return correct, total

def ACC_lf_total_match(A, Q, ACC_lf_correct):
    condiction = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', \
                    'price', 'num_connections', 'airline_preference']
    Q = Q[38:].split('AND')
    A = A[38:].split('AND')
    
    # for c in range(12):
    #     correct = 1
    #     Q_c = 0
    #     for i in range(c):
    #         if Q_c == len(Q):
    #             break
    #         if (condiction[i] in Q[Q_c]):
    #             Q_c += 1
    #             if (Q[Q_c] not in A):
    #                 correct = 0
    #     ACC_lf_correct[c] += correct
    for c in range(1, 13):
        correct = 1
        A_c = 0
        for i in range(c):
            if A_c == len(A):
                break
            # print('c : ', c, 'condiction[i] : ', condiction[i], 'A[A_c] : ', A[A_c], A_c)
            if (condiction[i] in A[A_c]):
                # print('A[A_c] : ', A[A_c], A_c)
                if (A[A_c] not in Q):
                    correct = 0
                A_c += 1
        if correct == 1:
            Q_c = 0
            for i in range(c):
                if Q_c == len(Q):
                    break
                if (condiction[i] in Q[Q_c]):
                    if (Q[Q_c] not in A):
                        correct = 0
                    Q_c += 1
        if c == 11 and correct == 1 and list(str(Q)) != list(str(A)):
            print('str Q : ', str(Q))
            print('str A : ', str(A))
            raise
        ACC_lf_correct[c-1] += correct
    return ACC_lf_correct

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

def QUERY_Output_predicted(cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, size_dialogue, state_tracking_label, truth_seq, SQL_YN, fp, args=None):
    
    ACC_lf_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ACC_lf_total = 0

    sql_fp, predict_fp, gt_fp, gate_fp, simple_fp, p_simple_fp, g_simple_fp = fp

    total_gate = 0
    correct_gate = 0

    total_query = 0
    correct_query = 0

    total_sql = 0
    total_correct_sql = 0

    sigm = nn.Sigmoid()
    cond_col_prob = sigm(cond_col_score)
    sigmoid_matrix = torch.ones_like(cond_col_prob) * 0.5
    predicted_cond_col = torch.gt(cond_col_prob, sigmoid_matrix).type(torch.cuda.FloatTensor)
    cond_gate_prob = sigm(cond_gate_score)
    sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
    predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor)

    batch_size = predicted_gate.size(0)
    # print(predicted_gate[:])
    # print(predicted_cond_col[:, -1])
    # print(size_dialogue)
    # print('cond_str_out : ', cond_str_out[0].size())
    # print('cond_op_score[i][ready_state_index[i]] : ', cond_op_score[0][[0, 2, 3, 5]].size())
    
    for i in range(batch_size):
        
        # last_index = search_t2(state_tracking_label[i]) # first gate
        last_index = search_t2(predicted_gate[i])

        # print('last : ', last_index)
        # truth_seq[i] = truth_seq[i].replace('LTEQL', '<=').replace('EQL', '=')
        # print('Truth : ', truth_seq[i])
        output_query = 'SELECT * FROM Airdialogue_Table WHERE '
        if float(SQL_YN[i]) == predicted_gate[i, last_index]:
            correct_gate += 1
            total_gate += 1
        else:
            total_gate += 1

        truth_seq_i = unicodedata.normalize('NFKD', truth_seq[i]).encode('ascii','ignore')
        true_query_simple = translate_query_to_simple(truth_seq_i[38:].split('AND'))
        # print('true_query_simple : ', true_query_simple)
        # print('truth_seq_i : ', truth_seq_i[38:].split('AND'))
        gt_fp.write(str(SQL_YN[i].data.item()) + ' | ' + truth_seq_i + '\n')
        gate_fp.write(str(predicted_gate[i, last_index].data.item()) + '\n')
        true_query_simple_str = [str(w) for w in true_query_simple]
        g_simple_fp.write(str(int(SQL_YN[i].data.item())) + ' | ' + " ".join(true_query_simple_str) + '\n')
        # gate_fp.write(str(predicted_gate[i].data.tolist()) + '\n')

        p_simple_str = '-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'
        
        if predicted_gate[i, last_index] == 1: # <t2> : gate open
            # print('Gate open !')
            col_list = predicted_cond_col[i, last_index]
            pred = ''
            for c in range(11):
                if col_list[c] == 1:
                    _, prediction = torch.max(cond_str_out[c][i][last_index].unsqueeze(0).data, 1)
                    pred += str(prediction.item()) + ' '
                else:
                    pred += str(-1) + ' '
            if col_list[11] == 1:
                pred += str(0) + ' '
            else:
                pred += str(-1) + ' '

            simple_fp.write(str(pred) + '\n')
            p_simple_str = pred

            if col_list[0] == 1:
                _, departure_airport = torch.max(cond_str_out[0][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + "departure_airport" 
                _, op = torch.max(cond_op_score[i][last_index][0].unsqueeze(0).data, 1)
                
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(departure_airport.item()) 
            if col_list[1] == 1:
                _, return_airport = torch.max(cond_str_out[1][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND return_airport" 
                _, op = torch.max(cond_op_score[i][last_index][1].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(return_airport.item()) 
            if col_list[2] == 1:
                _, departure_month = torch.max(cond_str_out[2][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND departure_month" 
                _, op = torch.max(cond_op_score[i][last_index][2].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(departure_month.item()) 
            if col_list[3] == 1:
                _, return_month = torch.max(cond_str_out[3][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND return_month" 
                _, op = torch.max(cond_op_score[i][last_index][3].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(return_month.item()) 
            if col_list[4] == 1:
                _, departure_day = torch.max(cond_str_out[4][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND departure_day" 
                _, op = torch.max(cond_op_score[i][last_index][4].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(departure_day.item())
            if col_list[5] == 1:
                _, return_day = torch.max(cond_str_out[5][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND return_day" 
                _, op = torch.max(cond_op_score[i][last_index][5].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(return_day.item())
            if col_list[6] == 1:
                _, departure_time_num = torch.max(cond_str_out[6][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND departure_time_num" 
                _, op = torch.max(cond_op_score[i][last_index][6].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(departure_time_num.item()) 
            if col_list[7] == 1:
                _, return_time_num = torch.max(cond_str_out[7][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND return_time_num" 
                _, op = torch.max(cond_op_score[i][last_index][7].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(return_time_num.item()) 
            if col_list[8] == 1:
                _, class_ = torch.max(cond_str_out[8][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND class" 
                _, op = torch.max(cond_op_score[i][last_index][8].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(class_.item()) 
            if col_list[9] == 1:
                _, price = torch.max(cond_str_out[9][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND price" 
                _, op = torch.max(cond_op_score[i][last_index][9].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(price.item())
            if col_list[10] == 1:
                _, num_connections = torch.max(cond_str_out[10][i][last_index].unsqueeze(0).data, 1)
                output_query = output_query + " AND num_connections" 
                _, op = torch.max(cond_op_score[i][last_index][10].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(num_connections.item())
            if col_list[11] == 1:
                output_query = output_query + " AND airline_preference" 
                _, op = torch.max(cond_op_score[i][last_index][11].unsqueeze(0).data, 1)
                if op.item() == 0:
                    output_query = output_query + " = "
                if op.item() == 1:
                    output_query = output_query + " <= "
                output_query = output_query + str(0) 
            # print('Query : ', output_query)
            sql_fp.write('Q : ' + truth_seq_i+'\n')
            sql_fp.write('A : ' + output_query+'\n\n\n')
            predict_fp.write(output_query+'\n')
            each_query_correct, each_query_total = total_match(truth_seq_i, output_query)
            
            total_query += each_query_total
            correct_query += each_query_correct

            if list(str(output_query)) != list(str(truth_seq_i)):
                # print(output_query)
                # print(truth_seq[i])
                # print('error !')
                # raise
                # print(error)
                # break
                total_sql += 1
            else:
                total_correct_sql +=1
                total_sql += 1

            # ACC_lf
            if SQL_YN[i] == 1:
                ACC_lf_total += 1
                ACC_lf_correct = ACC_lf_total_match(truth_seq_i, output_query, ACC_lf_correct)
        else:
            output_query = '0'
            sql_fp.write('Q : ' + output_query+'\n')
            sql_fp.write('A : ' + str(SQL_YN[i].data.item())+'\n\n\n')
            predict_fp.write(str(predicted_gate[i, last_index].data.item()) + '\n')
            simple_fp.write(str(int(predicted_gate[i, last_index].data.item())) + '\n')
            # print('Gate close !')
            # if SQL_YN[i] != 0:
            #     print('error !')
                # raise
                # print(error)
                # break
            # ACC_lf
            if SQL_YN[i] == 1:
                ACC_lf_total += 1
        p_simple_fp.write(str(int(predicted_gate[i, last_index].data.item())) + ' | ' + p_simple_str + '\n')


    return [correct_gate, total_gate] , [correct_query, total_query], [total_correct_sql, total_sql], ACC_lf_correct, ACC_lf_total
    
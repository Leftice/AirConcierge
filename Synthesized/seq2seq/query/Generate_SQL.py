from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch.autograd import Variable
import math
import unicodedata

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

def Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, truth_seq, SQL_YN, fp, args=None):
    
    t_fp, p_fp, sql_fp = fp
    batch_size = predicted_gate.size(0)

    sigm = nn.Sigmoid()
    cond_col_prob = sigm(cond_col_score)
    sigmoid_matrix = torch.ones_like(cond_col_prob) * 0.5
    predicted_cond_col = torch.gt(cond_col_prob, sigmoid_matrix).type(torch.cuda.FloatTensor)

    for i in range(batch_size):
        
        last_index = predicted_gate[i].size(0) - 1

        output_query = 'None'
        pred = 'None'

        # # Write ground truth answer
        # truth_seq_i = unicodedata.normalize('NFKD', truth_seq[i]).encode('ascii','ignore')
        # simple_truth_sql = translate_query_to_simple(truth_seq_i)
        # t_fp.write(str(SQL_YN[i].data.item()) + ' | ' + truth_seq_i + ' | ' + str(simple_truth_sql) + '\n')

        if predicted_gate[i, last_index] == 1: # <t2> : gate open
            col_list = predicted_cond_col[i, last_index]
            output_query = 'SELECT * FROM Airdialogue_Table WHERE '
            pred = ''
            for c in range(11):
                if col_list[c] == 1:
                    _, prediction = torch.max(cond_str_out[c][i][last_index].unsqueeze(0).data, 1)
                    pred += str(prediction.item()) + ' '
                else:
                    pred += str(-1) + ' ' # None
            if col_list[11] == 1:
                pred += str(0) + ' '
            else:
                pred += str(-1) + ' '

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

        # p_fp.write(str(int(predicted_gate[i, last_index].data.item())) + ' | ' + output_query + ' | ' + str(pred) + '\n')
        # sql_fp.write(output_query + '\n')

    return output_query, str(pred), str(int(predicted_gate[i, last_index].data.item()))
    
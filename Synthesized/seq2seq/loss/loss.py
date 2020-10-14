from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch.autograd import Variable
import math

loss_action =  nn.CrossEntropyLoss()
loss =  nn.CrossEntropyLoss(reduction='none')
CE = nn.CrossEntropyLoss(reduction='none')
log_softmax = nn.LogSoftmax()
bce_logit = nn.BCEWithLogitsLoss()

def compute_loss(logits1, logits2, logits3_rnn, target_diag, size_dialogue, action, size_action, mask1, mask2):
  
    # print('_compute_loss ...') # Input: (N, C), Target: (N)
    batch_size, length, vocab_size = logits1.size(0), logits1.size(1), logits1.size(2)
    logits1 = logits1.view(batch_size * length, -1)
    logits2 = logits2.view(batch_size * length, -1)
    batch_size_action, length_action, vocab_size_action = logits3_rnn.size(0), logits3_rnn.size(1), logits3_rnn.size(2)

    logits3_rnn_flatten = logits3_rnn.view(batch_size_action * length_action, -1)

    # print('target_diag : ', target_diag)
    # print('logits1 : ', logits1)

    # target_diag = target_diag[:, :torch.max(size_dialogue)]
    # mask1 = mask1[:, :torch.max(size_dialogue)]
    # mask2 = mask2[:, :torch.max(size_dialogue)]
    # print('target_diag : ', target_diag.size())
    target_diag = target_diag.reshape(batch_size * length, -1).squeeze(1)
    target_action = action.reshape(batch_size_action * length_action, -1).squeeze(1)
    mask1 = mask1.reshape(batch_size * length, ).type(torch.cuda.FloatTensor)
    mask2 = mask2.reshape(batch_size * length, ).type(torch.cuda.FloatTensor)
    # print('target_diag : ', target_diag.size())
    # print('logits1 : ', logits1.size())
    # print('logits2 : ', logits2.size())
    # print('mask1 : ', mask1.size())
    # print('mask2 : ', mask2.size())
    # print('target_diag : ', target_diag.size())p
    # print('action : ', action)
    # for item in logits3:
    #   print('logits3 : ', item)
      # print('logits3 : ', item.size())
    
    loss1_logits1 = loss(logits1, target_diag) 
    loss2_logits2 = loss(logits2, target_diag)
    loss3_logits3_rnn_loss = loss(logits3_rnn_flatten, target_action)
    # loss1_logits1 = F.nll_loss(F.log_softmax(logits1, dim=1), target_diag, reduction='none')
    # loss2_logits2 = F.nll_loss(F.log_softmax(logits2, dim=1), target_diag, reduction='none')

    loss1_logits1_mask = torch.mul(loss1_logits1, mask1)
    loss2_logits2_mask = torch.mul(loss2_logits2, mask2)
    loss1_logits1_mask = torch.sum(loss1_logits1_mask) / batch_size
    loss2_logits2_mask = torch.sum(loss2_logits2_mask) / batch_size
    loss3_logits3_rnn_loss = torch.sum(loss3_logits3_rnn_loss) / batch_size_action

    # print('logits3_rnn[:, 0] : ', logits3_rnn[:, 0].size())
    _, predicted_firstname_rnn = torch.max(F.softmax(logits3_rnn[:, 0], dim=1).data, 1)
    _, predicted_lastname_rnn = torch.max(F.softmax(logits3_rnn[:, 1], dim=1).data, 1)
    _, predicted_flight_rnn = torch.max(F.softmax(logits3_rnn[:, 2], dim=1).data, 1)
    _, predicted_state_rnn = torch.max(F.softmax(logits3_rnn[:, 3], dim=1).data, 1)

    correct_firstname_rnn = predicted_firstname_rnn.eq(action[:,0].data).cpu().sum()
    correct_lastname_rnn = predicted_lastname_rnn.eq(action[:,1].data).cpu().sum()
    correct_flight_rnn = predicted_flight_rnn.eq(action[:,2].data).cpu().sum()
    correct_state_rnn = predicted_state_rnn.eq(action[:,3].data).cpu().sum()

    # loss_firstname = self.loss_action(logits3[0], action[:,0])
    # # loss_firstname = torch.sum(loss_firstname) / batch_size
    # loss_lastname = self.loss_action(logits3[1], action[:,1]) # F.nll_loss(F.log_softmax(logits3[1], dim=1), action[:,1], reduction='none') # self.loss(logits3[1], action[:,1])
    # # loss_lastname = torch.sum(loss_lastname) / batch_size
    # loss_flight = self.loss_action(logits3[2], action[:,2]) # F.nll_loss(F.log_softmax(logits3[2], dim=1), action[:,2], reduction='none') # self.loss(logits3[2], action[:,2])
    # # loss_flight = torch.sum(loss_flight) / batch_size
    # loss_state = self.loss_action(logits3[3], action[:,3]) #F.nll_loss(F.log_softmax(logits3[3], dim=1), action[:,3], reduction='none') # self.loss(logits3[3], action[:,3])
    # # loss_state = torch.sum(loss_state) / batch_size

    # _, predicted_firstname = torch.max(F.softmax(logits3[0], dim=1).data, 1)
    # _, predicted_lastname = torch.max(F.softmax(logits3[1], dim=1).data, 1)
    # _, predicted_flight = torch.max(F.softmax(logits3[2], dim=1).data, 1)
    # _, predicted_state = torch.max(F.softmax(logits3[3], dim=1).data, 1)

    # correct_firstname = predicted_firstname.eq(action[:,0].data).cpu().sum()
    # correct_lastname = predicted_lastname.eq(action[:,1].data).cpu().sum()
    # correct_flight = predicted_flight.eq(action[:,2].data).cpu().sum()
    # correct_state = predicted_state.eq(action[:,3].data).cpu().sum()

    # print('loss_firstname : ', loss_firstname.size())
    # print('loss_lastname : ', loss_lastname.size())
    # print('loss_flight : ', loss_flight.size())
    # print('loss_state : ', loss_state.size())

    # return loss1_logits1_mask + loss2_logits2_mask + loss_firstname + loss_lastname + loss_flight + loss_state, [loss1_logits1_mask, loss2_logits2_mask, loss_firstname, loss_lastname, loss_flight, loss_state], [correct_firstname, correct_lastname, correct_flight, correct_state] 
    return loss1_logits1_mask + loss2_logits2_mask + loss3_logits3_rnn_loss, [loss1_logits1_mask, loss2_logits2_mask, loss3_logits3_rnn_loss], [correct_firstname_rnn, correct_lastname_rnn, correct_flight_rnn, correct_state_rnn] 

def compute_loss_nn(logits1, logits2, logits3, target_diag, size_dialogue, action, size_action, mask1, mask2, kb_true_answer_onehot, args, acc_print=True):
  
    # print('_compute_loss ...') # Input: (N, C), Target: (N)
    batch_size, length, vocab_size = logits1.size(0), logits1.size(1), logits1.size(2)
    logits1 = logits1.view(batch_size * length, -1)
    logits2 = logits2.view(batch_size * length, -1)

    target_diag = target_diag.reshape(batch_size * length, -1).squeeze(1)
    mask1 = mask1.reshape(batch_size * length, ).type(torch.cuda.FloatTensor)
    mask2 = mask2.reshape(batch_size * length, ).type(torch.cuda.FloatTensor)
    
    loss1_logits1 = loss(logits1, target_diag) 
    loss2_logits2 = loss(logits2, target_diag)

    loss1_logits1_mask = torch.mul(loss1_logits1, mask1)
    loss2_logits2_mask = torch.mul(loss2_logits2, mask2)
    loss1_logits1_mask = torch.sum(loss1_logits1_mask) / batch_size
    loss2_logits2_mask = torch.sum(loss2_logits2_mask) / batch_size

    loss_firstname = loss_action(logits3[0], action[:,0])
    loss_lastname = loss_action(logits3[1], action[:,1])
    loss_flight = loss_action(logits3[2], action[:,2])
    loss_state = loss_action(logits3[3], action[:,3])

    # correct_firstname, correct_lastname, correct_flight, correct_state = 0, 0, 0, 0
    _, predicted_firstname = torch.max(F.softmax(logits3[0], dim=1).data, 1)
    _, predicted_lastname = torch.max(F.softmax(logits3[1], dim=1).data, 1)
    _, predicted_flight = torch.max(F.softmax(logits3[2], dim=1).data, 1)
    _, predicted_state = torch.max(F.softmax(logits3[3], dim=1).data, 1)

    correct_firstname = predicted_firstname.eq(action[:,0].data).cpu().sum()
    correct_lastname = predicted_lastname.eq(action[:,1].data).cpu().sum()
    correct_flight = predicted_flight.eq(action[:,2].data).cpu().sum()
    correct_state = predicted_state.eq(action[:,3].data).cpu().sum()

    # # kb flight loss    
    # if args.sigmkb:
    #     correct_flight = 0
    #     kb_true_answer_onehot = kb_true_answer_onehot.type(torch.cuda.FloatTensor)
    #     sigm = nn.Sigmoid()
    #     flight_logit = sigm(logits3[2])
    #     loss_flight = -(args.sigmkb_alpha*(kb_true_answer_onehot * torch.log(flight_logit+1e-10)) + (1-kb_true_answer_onehot) * torch.log(1-flight_logit+1e-10))
    #     loss_flight = torch.mean(loss_flight)
    #     sigmoid_matrix = torch.ones_like(flight_logit) * 0.5
    #     predicted = torch.gt(flight_logit, sigmoid_matrix).type(torch.cuda.FloatTensor)
    #     # correct_flight = predicted.eq(kb_true_answer_onehot.data).cpu().sum()
    #     # _, true_flight = torch.max(kb_true_answer_onehot, 1)
    #     for b in range(flight_logit.size(0)):
    #         if torch.sum(predicted[b, :]) == 0 and torch.sum(kb_true_answer_onehot[b, :]) == 0:
    #            correct_flight += 1
    #         elif torch.sum(predicted[b, :]) != 0 and torch.sum(kb_true_answer_onehot[b, :]) != 0:
    #             _, pf = torch.max(flight_logit[b, :].unsqueeze(0).data, 1)
    #             _, tf = torch.max(kb_true_answer_onehot[b, :].unsqueeze(0), 1)
    #             # print('predict : ', pf)
    #             # print('answer : ', tf)
    #             correct_flight += pf.eq(tf.data).cpu().sum()

    loss3_logits3_loss = loss_firstname + loss_lastname + loss_flight + loss_state
    return loss1_logits1_mask + loss2_logits2_mask + loss3_logits3_loss, [loss1_logits1_mask, loss2_logits2_mask, loss3_logits3_loss], [correct_firstname, correct_lastname, correct_flight, correct_state] 

def compute_action_nn(logits3, action):
    # correct_firstname, correct_lastname, correct_flight, correct_state = 0, 0, 0, 0
    _, predicted_firstname = torch.max(F.softmax(logits3[0], dim=1).data, 1)
    _, predicted_lastname = torch.max(F.softmax(logits3[1], dim=1).data, 1)
    _, predicted_flight = torch.max(F.softmax(logits3[2], dim=1).data, 1)
    _, predicted_state = torch.max(F.softmax(logits3[3], dim=1).data, 1)

    correct_firstname = predicted_firstname.eq(action[:,0].data).cpu().sum()
    correct_lastname = predicted_lastname.eq(action[:,1].data).cpu().sum()
    correct_flight = predicted_flight.eq(action[:,2].data).cpu().sum()
    correct_state = predicted_state.eq(action[:,3].data).cpu().sum()
    return [correct_firstname, correct_lastname, correct_flight, correct_state], [predicted_firstname, predicted_lastname, predicted_flight, predicted_state]

def compute_att_loss(action_flight, kb_true_answer_onehot, kb_true_answer, args):
    criterion_bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    correct_flight = [0, 0]
    total_flight = [0, 0]
    loss_flight = ce(action_flight, kb_true_answer)
    _, predicted_flight = torch.max(F.softmax(action_flight, dim=1).data, 1)
    b = kb_true_answer_onehot.size(0)
    # print('predicted_flight[i, :] : ', predicted_flight.size())
    # print('predicted_flight[i, :] : ', predicted_flight[0])
    # print('kb_true_answer[i, :] : ', kb_true_answer[0])

    for i in range(b):
        if predicted_flight[i] == 30 and kb_true_answer[i] == 30:
            correct_flight[0] += 1
            total_flight[0] += 1
        elif predicted_flight[i] != 30 and kb_true_answer[i] == 30:
            total_flight[0] += 1
        elif predicted_flight[i] == 30 and kb_true_answer[i] != 30:
            total_flight[1] += 1
        elif predicted_flight[i] != 30 and kb_true_answer[i] != 30:
            _, pf = torch.max(action_flight[i, :].unsqueeze(0).data, 1)
            correct_flight[1] += pf.eq(kb_true_answer[i].data).cpu().sum()
            total_flight[1] += 1

    # correct_flight = [0, 0]
    # total_flight = [0, 0]
    # kb_true_answer_onehot = kb_true_answer_onehot.type(torch.cuda.FloatTensor)
    # sigm = nn.Sigmoid()
    # flight_logit = sigm(action_flight)
    # loss_flight = -(args.sigmkb_alpha*(kb_true_answer_onehot * torch.log(flight_logit+1e-10)) + (1-kb_true_answer_onehot) * torch.log(1-flight_logit+1e-10))
    # loss_flight = torch.sum(loss_flight, 1)
    # loss_flight = loss_flight.mean()
    # sigmoid_matrix = torch.ones_like(flight_logit) * 0.5
    # predicted = torch.gt(flight_logit, sigmoid_matrix).type(torch.cuda.FloatTensor)
    # # correct_flight = predicted.eq(kb_true_answer_onehot.data).cpu().sum()
    # # _, true_flight = torch.max(kb_true_answer_onehot, 1)
    # for b in range(flight_logit.size(0)):
    #     if torch.sum(predicted[b, :]) == 0 and torch.sum(kb_true_answer_onehot[b, :]) == 0:
    #         correct_flight[0] += 1
    #         total_flight[0] += 1
    #     elif torch.sum(predicted[b, :]) == 0 and torch.sum(kb_true_answer_onehot[b, :]) != 0:
    #         total_flight[1] += 1
    #     elif torch.sum(predicted[b, :]) != 0 and torch.sum(kb_true_answer_onehot[b, :]) == 0:
    #         total_flight[0] += 1
    #     elif torch.sum(predicted[b, :]) != 0 and torch.sum(kb_true_answer_onehot[b, :]) != 0:
    #         _, pf = torch.max(flight_logit[b, :].unsqueeze(0).data, 1)
    #         _, tf = torch.max(kb_true_answer_onehot[b, :].unsqueeze(0), 1)
    #         correct_flight[1] += pf.eq(tf.data).cpu().sum()
    #         total_flight[1] += 1
    return loss_flight, correct_flight, total_flight

def compute_sql_loss(truth_num, cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, state_tracking_label, SQL_YN, gate_lable, state_tracking_list, acc_print=True, args=None):
        loss = 0
        B = len(truth_num)

        ready_state_index = []
        for i in range(B):
            ready_state_index_each = []
            for j in range(state_tracking_label.size(1)):
                if state_tracking_label[i][j] == 1:
                    ready_state_index_each.append(j)
            ready_state_index.append(ready_state_index_each)

        # Evaluate the number of conditions
        cond_num_truth = map(lambda x:x[0], truth_num) # | truth_num : (len(sql['sql']['conds']), x[0]) column list , (x[1]) operator list)
        # print ("cond_num_truth", cond_num_truth)
        data_num = torch.from_numpy(np.array(cond_num_truth)).type(torch.LongTensor) # CE
        cond_num_truth_var = Variable(data_num.cuda())
        cond_num_score_loss, cond_num_score_correct, cond_num_score_total = compute_cond_num_score_loss(cond_num_score, cond_num_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print)

        # Evaluate the columns of conditions
        T = cond_col_score.size(2)
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][1]) > 0: # there are length = 0  | truth_num : (len(sql['sql']['conds']), x[0]) column list , (x[1]) operator list)
                truth_prob[b][list(truth_num[b][1])] = 1
        data_col = torch.from_numpy(truth_prob).type(torch.FloatTensor) # BCE
        cond_col_truth_var = Variable(data_col.cuda())
        sigm = nn.Sigmoid()
        cond_col_prob = sigm(cond_col_score)
        cond_col_score_loss, cond_col_score_correct, cond_col_score_total = compute_cond_col_score_loss(cond_col_prob, cond_col_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print)

        # Evaluate the operator of conditions
        T = cond_op_score.size(2)
        truth_op_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][1]) > 0: # there are length = 0  | truth_num : (len(sql['sql']['conds']), x[0]) column list , (x[1]) operator list)
                for col in range(len(truth_num[b][1])):
                    truth_op_prob[b][truth_num[b][1][col]] = truth_num[b][2][col]

        data_op = torch.from_numpy(truth_op_prob).type(torch.LongTensor) # CE
        cond_op_truth_var = Variable(data_op.cuda())
        cond_op_score_loss, cond_op_score_correct, cond_op_score_total = compute_cond_op_score_loss(cond_op_score, cond_op_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print)

        # Evaluate the classification string of conditions
        col_truth = map(lambda x:x[1], truth_num) # len, col_list, oper_list, string_list
        cond_str_truth = map(lambda x:x[3], truth_num) # | truth_num : (len(sql['sql']['conds']), x[0]) column list , (x[1]) operator list), (x[2]) string list)
        cond_str_out_loss, cond_str_out_correct, cond_str_out_total, cond_str_each = compute_cond_str_out_loss(cond_str_out, cond_str_truth, col_truth, state_tracking_label, SQL_YN, ready_state_index, acc_print)

        # Evaluate gate  
        sigm = nn.Sigmoid()
        cond_gate_prob = sigm(cond_gate_score)
        cond_gate_score_loss, cond_gate_score_correct, cond_gate_score_total = compute_cond_gate_score_loss(cond_gate_prob, gate_lable, acc_print)

        return cond_num_score_loss + cond_col_score_loss + cond_op_score_loss + args.sql_alpha2 * cond_str_out_loss + cond_gate_score_loss, [cond_num_score_loss, cond_col_score_loss, cond_op_score_loss, cond_str_out_loss, cond_gate_score_loss], \
                [cond_num_score_correct, cond_col_score_correct, cond_op_score_correct, cond_str_out_correct, cond_gate_score_correct], \
                [cond_num_score_total, cond_col_score_total, cond_op_score_total, cond_str_out_total, cond_gate_score_total], cond_str_each

def compute_cond_num_score_loss(cond_num_score, cond_num_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print):
    b = cond_num_score.size(0)
    loss = None
    total_example = 0

    correct = 0
    total = 0

    for i in range(b):
        if SQL_YN[i] == 0:
            continue
        each_num = torch.sum(state_tracking_label[i])
        total_example += each_num
        # ready_state_index = []
        # for j in range(state_tracking_label.size(1)):
        #     if state_tracking_label[i][j] == 1:
        #        ready_state_index.append(j) 
        cond_num_truth_var_each = cond_num_truth_var[i].expand(each_num, )
        if loss is None:
            loss = CE(cond_num_score[i][ready_state_index[i]], cond_num_truth_var_each)
            _, predicted = torch.max(cond_num_score[i][ready_state_index[i]].data, 1)
            total += cond_num_truth_var_each.size(0)
            correct += predicted.eq(cond_num_truth_var_each.data).cpu().sum()
            correct = correct.item()
        else:
            loss = torch.cat((loss, CE(cond_num_score[i][ready_state_index[i]], cond_num_truth_var_each)), dim=0)
            _, predicted = torch.max(cond_num_score[i][ready_state_index[i]].data, 1)
            total += cond_num_truth_var_each.size(0)
            correct += predicted.eq(cond_num_truth_var_each.data).cpu().sum()
            correct = correct.item()
    if loss is not None:
        total_size = loss.size(0)
        loss = torch.sum(loss) / total_size
    else:
        loss = 0
    return loss, correct, total

def compute_cond_col_score_loss(cond_col_prob, cond_col_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print):
    b = cond_col_prob.size(0)
    loss = None
    total_example = 0
    
    correct = 0
    total = 0

    for i in range(b):
        if SQL_YN[i] == 0:
            continue
        each_num = torch.sum(state_tracking_label[i])
        # ready_state_index = []
        # for j in range(state_tracking_label.size(1)):
        #     if state_tracking_label[i][j] == 1:
        #        ready_state_index.append(j) 
        cond_col_truth_var_each = cond_col_truth_var[i].expand(each_num, -1)
        if loss is None:
            each_bce_loss = -(3*(cond_col_truth_var_each * torch.log(cond_col_prob[i][ready_state_index[i]]+1e-10)) + (1-cond_col_truth_var_each) * torch.log(1-cond_col_prob[i][ready_state_index[i]]+1e-10))
            loss = each_bce_loss
            sigmoid_matrix = torch.ones_like(cond_col_prob[i][ready_state_index[i]]) * 0.5
            predicted = torch.gt(cond_col_prob[i][ready_state_index[i]], sigmoid_matrix).type(torch.cuda.FloatTensor)
            correct += predicted.eq(cond_col_truth_var_each.data).cpu().sum()
            correct = correct.item()
            total += cond_col_truth_var_each.size(0) * cond_col_truth_var_each.size(1)
        else:
            each_bce_loss = -(3*(cond_col_truth_var_each * torch.log(cond_col_prob[i][ready_state_index[i]]+1e-10)) + (1-cond_col_truth_var_each) * torch.log(1-cond_col_prob[i][ready_state_index[i]]+1e-10))
            loss = torch.cat((loss, each_bce_loss), dim=0)
            sigmoid_matrix = torch.ones_like(cond_col_prob[i][ready_state_index[i]]) * 0.5
            predicted = torch.gt(cond_col_prob[i][ready_state_index[i]], sigmoid_matrix).type(torch.cuda.FloatTensor)
            correct += predicted.eq(cond_col_truth_var_each.data).cpu().sum()
            correct = correct.item()
            total += cond_col_truth_var_each.size(0) * cond_col_truth_var_each.size(1)
    if loss is not None:
        loss = torch.mean(loss)
    else:
        loss = 0
    return loss, correct, total

def compute_cond_op_score_loss(cond_op_score, cond_op_truth_var, state_tracking_label, SQL_YN, ready_state_index, acc_print):
    b = cond_op_score.size(0)
    loss = None
    total_example = 0

    correct = 0
    total = 0
    
    for i in range(b):
        if SQL_YN[i] == 0:
            continue
        each_num = torch.sum(state_tracking_label[i])
        total_example += each_num
        # ready_state_index = []
        # for j in range(state_tracking_label.size(1)):
        #     if state_tracking_label[i][j] == 1:
        #        ready_state_index.append(j) 
        cond_op_truth_var_each = cond_op_truth_var[i].expand(each_num, -1, )
        temp_ready_state_score = cond_op_score[i][ready_state_index[i]]
        for k in range(each_num):
            if loss is None:
                loss = CE(temp_ready_state_score[k], cond_op_truth_var_each[k].type(torch.cuda.LongTensor))
                _, predicted = torch.max(temp_ready_state_score[k].data, 1)
                total += cond_op_truth_var_each[k].type(torch.cuda.LongTensor).size(0)
                correct += predicted.eq(cond_op_truth_var_each[k].type(torch.cuda.LongTensor).data).cpu().sum()
                correct = correct.item()
            else:
                loss = torch.cat((loss, CE(temp_ready_state_score[k], cond_op_truth_var_each[k].type(torch.cuda.LongTensor))), dim=0)
                _, predicted = torch.max(temp_ready_state_score[k].data, 1)
                total += cond_op_truth_var_each[k].type(torch.cuda.LongTensor).size(0)
                correct += predicted.eq(cond_op_truth_var_each[k].type(torch.cuda.LongTensor).data).cpu().sum()
                correct = correct.item()
    if loss is not None:
        total_size = loss.size(0)
        loss = torch.sum(loss) / total_size
    else:
        loss = 0
    return loss, correct, total

def compute_cond_str_out_loss(cond_str_out, cond_str_truth, col_truth, state_tracking_label, SQL_YN, ready_state_index, acc_print):
    b = cond_str_out[0].size(0)
    total_loss = 0
    num_of_col = 11 # without air prefer

    each_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    each_total = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    each_loss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    each_weight = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

    correct = 0
    total = 0

    for c in range(num_of_col):
        loss = None
        total_example = 0
        for i in range(b):
            if SQL_YN[i] == 0:
                continue
            if c not in list(col_truth[i]):
                continue
            curr_index = list(col_truth[i]).index(c)
            data_str = torch.from_numpy(np.array(cond_str_truth[i][curr_index])).type(torch.LongTensor) # CE
            cond_str_truth_var = Variable(data_str.cuda())
            each_num = torch.sum(state_tracking_label[i])
            total_example += each_num
            # ready_state_index = []
            # for j in range(state_tracking_label.size(1)):
            #     if state_tracking_label[i][j] == 1:
            #        ready_state_index.append(j) 
            cond_str_truth_var_each = cond_str_truth_var.expand(each_num, )
            if loss is None:
                loss = CE(cond_str_out[c][i][ready_state_index[i]], cond_str_truth_var_each)
                _, predicted = torch.max(cond_str_out[c][i][ready_state_index[i]].data, 1)
                # print('c : ', c)
                # print('ready_state_index[i] : ', ready_state_index[i])
                # print('str predicted : ', predicted)
                # print('cond_str_truth_var_each : ', cond_str_truth_var_each)
                # raise
                total += cond_str_truth_var_each.size(0); each_total[c] += cond_str_truth_var_each.size(0)
                correct += predicted.eq(cond_str_truth_var_each.data).cpu().sum(); each_correct[c] += predicted.eq(cond_str_truth_var_each.data).cpu().sum()
                correct = correct.item(); each_correct[c] = each_correct[c].item()
            else:
                loss = torch.cat((loss, CE(cond_str_out[c][i][ready_state_index[i]], cond_str_truth_var_each)), dim=0)
                _, predicted = torch.max(cond_str_out[c][i][ready_state_index[i]].data, 1)
                total += cond_str_truth_var_each.size(0); each_total[c] += cond_str_truth_var_each.size(0)
                correct += predicted.eq(cond_str_truth_var_each.data).cpu().sum(); each_correct[c] += predicted.eq(cond_str_truth_var_each.data).cpu().sum()
                correct = correct.item(); each_correct[c] = each_correct[c].item()

        if loss is not None:
            total_size = loss.size(0)
            loss = torch.sum(loss) / total_size
            total_loss += each_weight[c] * loss; each_loss[c] += loss
        else:
            total_loss += each_weight[c] * 0; each_loss[c] += 0

    for c in range(num_of_col):
        if each_total[c] > 1:
            each_total[c] = each_total[c]-1
    return total_loss, correct, total, [each_loss, each_correct, each_total]

def compute_cond_gate_score_loss(cond_gate_prob, gate_lable, acc_print):
    b, s = cond_gate_prob.size(0), cond_gate_prob.size(1)
    loss = None
    total_example = 0
    
    correct = 0
    total = 0
    
    gate_lable_var = gate_lable.type(torch.cuda.FloatTensor)
    loss = -(gate_lable_var * torch.log(cond_gate_prob+1e-10) + (1-gate_lable_var) * torch.log(1-cond_gate_prob+1e-10))
    if acc_print:        
        sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
        predicted = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor)
        correct += predicted.eq(gate_lable_var.data).cpu().sum()
        correct = correct.item()
        total += gate_lable_var.size(0) * gate_lable_var.size(1)
    if loss is not None:
        loss = torch.sum(loss) / total
    else:
        loss = 0

    return loss, correct, total
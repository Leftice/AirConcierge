from __future__ import division
import logging
import os
import random
import time

import torch
# import torchtext
from torch import optim

import seq2seq
from seq2seq.loss import *
from seq2seq.query import *
from seq2seq.database import *
from seq2seq.util.checkpoint import Checkpoint
from utils.utils import *
from tensorboardX import SummaryWriter 
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence

class SupervisedSelfPlayEval(object):
    def __init__(self, model_dir='checkpoints/', args=None, corpus=None):
        self._trainer = "Simple Inference"

        if not os.path.exists(model_dir):
            print('No such model dir !')
            raise
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.corpus = corpus

    def list_to_tensor(self, ll):
        tensor_list = []
        for i in ll:
            tensor_list.append(torch.tensor(np.array(i).astype(int)))
        return tensor_list

    def check_boundary(self, dialogue):
        t1_list = []
        t2_list = []
        index = 0
        for token in dialogue:
            if token == '<t1>':
                t1_list.append(str(index))
            if token == '<t2>':
                t2_list.append(str(index))
            if token == '<eod>':
                break
            index = index + 1
        return t1_list + t2_list

    def _test_epoches_t1t2_self_play_eval(self, dataloader, model, start_step, save_dir='runs/exp', args=None):

        if args.syn:
            data_path = './results/synthesized/'
        elif args.air:
            data_path = './results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        model.eval()
        total_step = start_step
        if not os.path.exists(data_path + 'SelfPlay_Eval/Result/'):
            os.makedirs(data_path + 'SelfPlay_Eval/Result/')
        t_fp = open(data_path + 'SelfPlay_Eval/Result/self_output.data', 'w')
        a_fp = open(data_path + 'SelfPlay_Eval/Result/self_output_action.data', 'w')
        log_fp = open(data_path + 'selfplay_eval_output.log', 'w')

        with torch.no_grad():
            
            train_loss = 0
            total = 0
            correct_firstname, correct_lastname, correct_flight, correct_state, correct_flight2 = 0, 0, 0, 0, 0
            all_combination_correct = [0, 0, 0, 0, 0]
            all_combination_total = [0, 0, 0, 0, 0]
            all_combination_name = ['<st_book>', '<st_change>', '<st_cancel>', '<st_no_reservation>', '<st_no_flight>']
            
            for batch_idx, (intent, size_intent, action, kb, has_reservation, col_seq, truth_seq, SQL_YN, turn_gate) in enumerate(dataloader):
                
                intent, size_intent, action, kb, has_reservation, col_seq, SQL_YN = intent.cuda(), size_intent.cuda(), action.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda(), SQL_YN.cuda()
                
                # intent and action 
                b = intent.size(0)
                intent_sents = [[] for _ in range(b)]
                action_sents = [[] for _ in range(b)]
                action_sentsw = [[] for _ in range(b)]
                for i in range(b):
                    for j in intent[i]:
                        intent_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                    for j in range(2):
                        action_sents[i].append(self.corpus.dictionary.idx2word[action[i,j].data])
                        action_sentsw[i].append(self.corpus.dictionary.idx2word[action[i,j].data])
                    action_sents[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                    if int(action[i,2].data) == 30:
                        action_sentsw[i].append('<fl_empty>')
                    else:
                        action_sentsw[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                    action_sents[i].append(self.corpus.dictionary.idx2word[action[i,3].data])
                    action_sentsw[i].append(self.corpus.dictionary.idx2word[action[i,3].data])
                
                history = [[1]] # <t1>
                history_sents = [['<t1>']]
                SQL_buffer = []
                request = 0
                DB_kb_buffer = []
                eod_id= self.corpus.dictionary.word2idx['<eod>']
                end = 0
                concat_flight_embed = torch.zeros((1, 1, 256)); concat_flight_embed = concat_flight_embed.cuda()
                turn_index = -1
                predict_flight_number = 'None'

                for turn_i in range(args.max_dialogue_turns):

                    # Customer
                    if turn_i % 2 == 0: # <t1>
                        if args.sql:
                            # list to tensor
                            source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                            b = source_diag.size(0)
                            size_history = []
                            for i in range(b):
                                size_history.append(len(history[i]))
                            size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                            # display source input 
                            source_diag_sents = [[] for _ in range(b)]
                            for i in range(b):
                                for j in source_diag[i]:
                                    source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                            logits_train1, sequence_symbols = model.module.Call_t1_SelfPlayEval(intent, size_intent, source_diag, size_dialogue, args=args)
                            
                        # ('sequence_symbols : ', sequence_symbols)
                        sents = [[] for _ in range(b)]
                        length = [0 for _ in range(b)]
                        for s in range(len(sequence_symbols)):
                            for i in range(b):
                                token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                                if token == '<t2>' and length[i] == 0:
                                    sents[i].append(token)
                                    length[i] = len(sents[i])
                                    history[i].append(sequence_symbols[s][i])
                                    history_sents[i].append(token)
                                elif token != '<t2>' and length[i] == 0:
                                    sents[i].append(token)
                                    history[i].append(sequence_symbols[s][i])
                                    history_sents[i].append(token)
                        for i in range(b):
                            if '<eod>' in sents[i]:
                                end = 1
                                boundary_list = self.check_boundary(history_sents[i])
                                out = " ".join(intent_sents[i]) + "|" + " ".join(action_sents[i]) + "|" + " ".join(history_sents[i]) + "|" + " ".join(boundary_list) + "\n"
                                t_fp.write(out)
                                break
                        
                    # Agent
                    if turn_i % 2 == 1: # <t2>
                        if args.sql:
                            # Dialogue input : list to tensor
                            source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                            b = source_diag.size(0)
                            size_history = []
                            for i in range(b):
                                size_history.append(len(history[i]))
                            size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                            # Display source input 
                            source_diag_sents = [[] for _ in range(b)]
                            for i in range(b):
                                for j in source_diag[i]:
                                    source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])

                            # Send query ? + Generate SQL 
                            if turn_i == turn_gate[0]:
                                request = 1
                                # print('initi kb : ', kb.size())
                                if kb[0].size(0) == 1 : # empty
                                    # print('kb : ', kb[0].size())
                                    # print('Empty')
                                    pass
                                elif kb[0].size(0) == 2 : # one flight
                                    # print('kb : ', kb[0].size())
                                    # print('One flight')
                                    concat_flight = kb[0, 0:1]
                                    # print('concat_flight : ', concat_flight.size())
                                    predict_flight_number = self.corpus.dictionary.idx2word[concat_flight[0, -1]]
                                    # print('predict_flight_number : ', predict_flight_number)
                                    concat_flight_embed = model.module.Encode_Flight_KB(concat_flight.unsqueeze(0))
                                else: # result > 1
                                    # print('More than one flight')
                                    global_pointer = model.module.Point_Encode_KB(source_diag, size_dialogue, kb, has_reservation, col_seq, args=args)
                                    # print('global_pointer : ', global_pointer)
                                    _, global_pointer_index = torch.max(F.softmax(global_pointer, dim=1).data, 1)
                                    # print('global_pointer_index : ', global_pointer_index)
                                    concat_flight = kb[0, global_pointer_index]
                                    # print('concat_flight : ', concat_flight.size(), concat_flight)
                                    predict_flight_number = self.corpus.dictionary.idx2word[concat_flight[0, -1]]
                                    # print('predict_flight_number : ', predict_flight_number)
                                    concat_flight_embed = model.module.Encode_Flight_KB(concat_flight.unsqueeze(0))
                                logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                                turn_index = predicted_gate[0].size(0) - 1
                            
                            if request == 1:
                                logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, turn_gate=turn_index, args=args)
                            else:
                                logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                        
                        # ('sequence_symbols : ', sequence_symbols)
                        sents = [[] for _ in range(b)]
                        length = [0 for _ in range(b)]
                        for s in range(len(sequence_symbols)):
                            for i in range(b):
                                token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                                if token == '<t1>' and length[i] == 0:
                                    sents[i].append(token)
                                    length[i] = len(sents[i])
                                    history[i].append(sequence_symbols[s][i])
                                    history_sents[i].append(token)
                                elif token != '<t1>' and length[i] == 0:
                                    sents[i].append(token)
                                    history[i].append(sequence_symbols[s][i])
                                    history_sents[i].append(token)
                        if args.air:
                            if turn_i == turn_gate[0]:
                                add_token = ['flight', 'number', 'is', predict_flight_number, '.', '<t1>']
                                if any('<fl_' in s for s in sents[0]) == False and '<fl_10' in predict_flight_number:
                                    history[i] = history[i][:-1] # pop out <t1>
                                    history_sents[i] = history_sents[i][:-1] # pop out <t1>
                                    for add_s in add_token:
                                        history[i].append(self.corpus.dictionary.word2idx[add_s])
                                        history_sents[i].append(add_s)

                        for i in range(b):
                            if '<eod>' in sents[i]:
                                end = 1
                                boundary_list = self.check_boundary(history_sents[i])
                                history_gate = []
                                for h in range(len(history_sents[i])):
                                    if h < predicted_gate[i].size(0):
                                        history_gate.append(str(history_sents[i][h]) + '_' + str(predicted_gate[i][h].data.item()))
                                    else:
                                        history_gate.append(history_sents[i][h])
                                out = " ".join(intent_sents[i]) + "|" + " ".join(action_sents[i]) + "|" + " ".join(history_gate) + "|" + " ".join(boundary_list) + "\n"
                                t_fp.write(out)
                                t_fp.flush()
                                break
                        
                    if end == 1:
                        break
                # action
                source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                b = source_diag.size(0)
                size_history = []
                for i in range(b):
                    size_history.append(len(history[i]))
                size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                # display source input 
                source_diag_sents = [[] for _ in range(b)]
                for i in range(b):
                    for j in source_diag[i]:
                        source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                logits_train3 = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, turn_gate=turn_index, end=1, args=args)
                
                correct, predict_action = compute_action_nn(logits_train3, action)
                correct_firstname += correct[0].item() ; 
                correct_lastname += correct[1].item() ;
                correct_flight += correct[2].item();
                correct_state += correct[3].item() ;
                predict_action_name1 = self.corpus.dictionary.idx2word[predict_action[0][0].data.item()]
                predict_action_name2 = self.corpus.dictionary.idx2word[predict_action[1][0].data.item()]
                predict_action_flight = predict_action[2][0].data.item()
                if predict_action_flight == 30:
                    predict_action_flight = '<fl_empty>'
                else:
                    predict_action_flight = '<fl_' + str(int(predict_action_flight)+1000) + '>'
                predict_action_state = self.corpus.dictionary.idx2word[predict_action[3][0].data.item()]
                action_out = " ".join(action_sentsw[i]) + "|" + str(predict_action_name1) + " " + str(predict_action_name2) + " " + str(predict_action_flight) + " " + str(predict_action_state) + "\n"
                a_fp.write(action_out)
                a_fp.flush()

                # check answer
                if action_sents[i][3] == '<st_book>':
                    all_combination_total[0] += 1
                    if action_sents[i][3] == predict_action_state:
                        all_combination_correct[0] += 1
                elif action_sents[i][3] == '<st_change>':
                    all_combination_total[1] += 1
                    if action_sents[i][3] == predict_action_state:
                        all_combination_correct[1] += 1
                elif action_sents[i][3] == '<st_cancel>':
                    all_combination_total[2] += 1
                    if action_sents[i][3] == predict_action_state:
                        all_combination_correct[2] += 1
                elif action_sents[i][3] == '<st_no_reservation>':
                    all_combination_total[3] += 1
                    if action_sents[i][3] == predict_action_state:
                        all_combination_correct[3] += 1
                elif action_sents[i][3] == '<st_no_flight>':
                    all_combination_total[4] += 1
                    if action_sents[i][3] == predict_action_state:
                        all_combination_correct[4] += 1
                
                # for s in range(len(all_combination_correct)):
                #     if all_combination_total[s] == 0:
                #         print(all_combination_name[s], ' : -- ')
                #     else:
                #         print(all_combination_name[s], ' : ', 100.*all_combination_correct[s] / all_combination_total[s], ' | ', all_combination_correct[s], ' / ', all_combination_total[s])

                total = total + 1
                progress_bar(batch_idx, len(dataloader), 'Acc : %.1f%% (%d/%d) |: %.1f%% (%d/%d) |: %.1f%% (%d/%d) |: %.1f%% (%d/%d)'
                     % (100. * correct_firstname / total, correct_firstname, total, \
                        100. * correct_lastname / total, correct_lastname, total, \
                        100. * correct_flight / total, correct_flight, total, \
                        100. * correct_state / total, correct_state, total))
            log_fp.write('Acc : %.1f%% (%d/%d) |: %.1f%% (%d/%d) |: %.1f%% (%d/%d) |: %.1f%% (%d/%d)'
                     % (100. * correct_firstname / total, correct_firstname, total, \
                        100. * correct_lastname / total, correct_lastname, total, \
                        100. * correct_flight / total, correct_flight, total, \
                        100. * correct_state / total, correct_state, total))

        t_fp.close()
        a_fp.close()
        log_fp.close()

    def test(self, args, model, dataloader, resume=False, save_dir='runs/exp'):

        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.model_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model.load_state_dict(resume_checkpoint.model)
            self.optimizer = None
            self.args = args
            model.args = args
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            print('Resume from ', latest_checkpoint_path)
            print('start_epoch : ', start_epoch)
            print('step : ', step)
            start_epoch = 1
            step = 0
        else:
            print('Please Resume !')
            raise
        self._test_epoches_t1t2_self_play_eval(dataloader, model, step, save_dir=save_dir, args=args)
        return model
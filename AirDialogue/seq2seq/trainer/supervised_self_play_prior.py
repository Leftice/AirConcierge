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

class SupervisedSelfPlayPrior(object):
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

    def translate_query_to_simple(self, query):
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
        if not os.path.exists(data_path + 'SelfPlay_Eval/Prior/'):
            os.makedirs(data_path + 'SelfPlay_Eval/Prior/')
        t_fp = open(data_path + 'SelfPlay_Eval/Prior/self_output.data', 'w')

        if not os.path.exists(data_path + 'SelfPlay_Eval/SQL/'):
            os.makedirs(data_path + 'SelfPlay_Eval/SQL/')
        gt_fp = open(data_path + 'SelfPlay_Eval/SQL/ground_truth_.txt', 'w')
        p_fp = open(data_path + 'SelfPlay_Eval/SQL/predicted_.txt', 'w')
        sql_fp = open(data_path + 'SelfPlay_Eval/SQL/predicted_SQL.txt', 'w')
        gate_fp = open(data_path + 'SelfPlay_Eval/SQL/prior_gate.txt', 'w')
        fp = [gt_fp, p_fp, sql_fp]

        with torch.no_grad():
            total = 0
            for batch_idx, (intent, size_intent, action, has_reservation, col_seq, truth_seq, SQL_YN, kb_true_answer) in enumerate(dataloader):
                
                intent, size_intent, action, has_reservation, col_seq, SQL_YN, kb_true_answer = intent.cuda(), size_intent.cuda(), action.cuda(), has_reservation.cuda(), col_seq.cuda(), SQL_YN.cuda(), kb_true_answer.cuda()
                
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
                # print('*'*100)
                # for i in range(b):
                #     print('Intent   : ', str(intent_sents[i]))
                #     print('Action  : ', str(action_sents[i]))
                # print('*'*100)
                
                history = [[1]] # <t1>
                history_sents = [['<t1>']]
                SQL_buffer = []
                request = 0
                DB_kb_buffer = []
                eod_id= self.corpus.dictionary.word2idx['<eod>']
                end = 0
                t2_gate_turn = -1
                SQL_query = 'None'
                pred = 'None'
                pred_gate = '0'

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
                                if token == '<t2>' and length[i] == 0: # end of sentence
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
                                t_fp.flush()
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
                            cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate = model.module.SQL_AND_StateTracking(source_diag, size_dialogue, col_seq, args=args)
                            request = predicted_gate[0][-1].data.item()
                            request_index = predicted_gate[0].size(0) - 1
                            if request == 1.:
                                SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, truth_seq, SQL_YN, fp, args=args)
                                t2_gate_turn = turn_i
                                end = 1
                                break
                            else:
                                concat_flight = torch.zeros((1, 1, 256)); concat_flight = concat_flight.cuda()
                                logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEvalPrior(source_diag, size_dialogue, has_reservation, col_seq, concat_flight, args=args)

                                ################################################################
                                ######################### Prior ################################
                                ################################################################
                                prior_history = [list(history[0])]
                                prior_sents = [[] for _ in range(b)]
                                prior_length = [0 for _ in range(b)]
                                for s in range(len(sequence_symbols)):
                                    for i in range(b):
                                        token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                                        if token == '<t1>' and prior_length[i] == 0:
                                            prior_sents[i].append(token)
                                            prior_length[i] = len(sents[i])
                                            prior_history[i].append(sequence_symbols[s][i])
                                        elif token != '<t1>' and prior_length[i] == 0:
                                            prior_sents[i].append(token)
                                            prior_history[i].append(sequence_symbols[s][i])

                                # print('*'*100)
                                # print('history : ', [self.corpus.dictionary.idx2word[w] for w in history[0]])
                                # print('prior_history : ', [self.corpus.dictionary.idx2word[w] for w in prior_history[0]])
                                # list to tensor
                                prior_source_diag = pad_sequence(self.list_to_tensor(prior_history), batch_first=True, padding_value=eod_id); prior_source_diag = prior_source_diag.cuda()
                                b = prior_source_diag.size(0)
                                prior_size_history = []
                                for i in range(b):
                                    prior_size_history.append(len(prior_history[i]))
                                prior_size_dialogue = torch.tensor(prior_size_history, dtype=torch.int64); prior_size_dialogue = prior_size_dialogue.cuda()
                                prior_predicted_gate = model.module.Prior_Gate(prior_source_diag, prior_size_dialogue, col_seq)
                                # # display source input 
                                # prior_source_diag_sents = [[] for _ in range(b)]
                                # for i in range(b):
                                #     for j in prior_source_diag[i]:
                                #         prior_source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                                
                                # history_prior_gate = []
                                # for h in range(len(prior_history[i])):
                                #     if h < prior_predicted_gate[i].size(0):
                                #         history_prior_gate.append(str(self.corpus.dictionary.idx2word[prior_history[i][h]]) + '_' + str(prior_predicted_gate[i][h].data.item()))
                                #     else:
                                #         history_prior_gate.append(history_sents[i][h])
                                # print('history_prior_gate : ', history_prior_gate)
                                # print('predicted_gate : ', predicted_gate)
                                # prior gate predicted

                                if torch.sum(prior_predicted_gate[i]).data.item() != 0 and request == 0:
                                    # print('predicted_gate : ', predicted_gate.size(), ' prior_predicted_gate : ', prior_predicted_gate.size(), ' request :', request, request_index)
                                    fix_predict_gate = predicted_gate
                                    fix_predict_gate[0][request_index] = prior_predicted_gate[0][request_index+1]
                                    predicted_gate = fix_predict_gate

                                    request = 1.
                                    request_index = predicted_gate[0].size(0) - 1
                                    SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, truth_seq, SQL_YN, fp, args=args)
                                    t2_gate_turn = turn_i
                                    end = 1

                                ################################################################
                                ######################### Prior ################################
                                ################################################################

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
                        
                        for i in range(b):
                            # print('History Now     : ', (history_sents[i]))
                            # history_gate = []
                            # for h in range(len(history_sents[i])):
                            #     if h < predicted_gate[i].size(0):
                            #         history_gate.append(str(history_sents[i][h]) + '_' + str(predicted_gate[i][h].data.item()))
                            #     else:
                            #         history_gate.append(history_sents[i][h])
                            # print('predicted_gate : ', history_gate)
                            # print('*'*100)
                            if '<eod>' in sents[i]:
                                # print('*'*100)
                                # print('Turn : ', turn_i)
                                # print('Intent   : ', (intent_sents[i]))
                                # print('Action  : ', (action_sents[i]))
                                # print('History Input   : ', (source_diag_sents[i]))
                                # print('Customer Output : ', (sents[i]))
                                # print('History Now     : ', (history_sents[i]))
                                # print('*'*100)
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

                # Write ground truth answer
                truth_seq_i = unicodedata.normalize('NFKD', truth_seq[0]).encode('ascii','ignore')
                simple_truth_sql = translate_query_to_simple(truth_seq_i)
                gt_fp.write(str(SQL_YN[0].data.item()) + ' | ' + truth_seq_i + ' | ' + str(simple_truth_sql) + '\n'); gt_fp.flush()
                p_fp.write(pred_gate + ' | ' + SQL_query + ' | ' + pred + '\n'); p_fp.flush()
                sql_fp.write(SQL_query + '\n'); sql_fp.flush()
                gate_fp.write(str(t2_gate_turn) + '\n'); gate_fp.flush()

                progress_bar(batch_idx, len(dataloader))
        t_fp.close()

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
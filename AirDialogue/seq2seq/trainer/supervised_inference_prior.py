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
from seq2seq.util.checkpoint import Checkpoint
from utils.utils import *
from tensorboardX import SummaryWriter 
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence

class SupervisedInferencePrior(object):
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

    def _test_epoches_t1t2(self, dataloader, model, start_step, save_dir='runs/exp', args=None):

        model.eval()
        total_step = start_step
        # if not os.path.exists('Inference_Bleu/t1t2/'):
        #     os.makedirs('Inference_Bleu/t1t2/')
        # t_fp = open('Inference_Bleu/t1t2/dev_inference_out.txt', 'w')
        # teacher_fp = open('Inference_Bleu/t1t2/dev_inference_out_teacher.txt', 'w')

        if args.syn:
            data_path = './results/synthesized/'
        elif args.air:
            data_path = './results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        if not os.path.exists(data_path + 'Inference_Bleu/SQL/'):
            os.makedirs(data_path + 'Inference_Bleu/SQL/')
        gt_fp = None
        p_fp = open(data_path + 'Inference_Bleu/SQL/predicted_.txt', 'w')
        sql_fp = open(data_path + 'Inference_Bleu/SQL/predicted_SQL.txt', 'w')
        gate_fp = open(data_path + 'Inference_Bleu/SQL/prior_gate.txt', 'w')
        fp = [gt_fp, p_fp, sql_fp]

        with torch.no_grad():
            train_loss = 0
            total = 0
            eod_id= self.corpus.dictionary.word2idx['<eod>']

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, kb_true_answer = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda(), kb_true_answer.cuda()

                SQL_query = 'None'
                pred = 'None'
                pred_gate = '0'
                t2_gate_turn_index = -1

                if source_diag[0, -1] == 1: # <t1>
                    total_step += 1

                if source_diag[0, -1] == 2: # <t2>
                    if args.sql:
                        b = source_diag.size(0)
                        # Send query ? + Generate SQL 
                        cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate = model.module.SQL_AND_StateTracking(source_diag, size_dialogue, col_seq, args=args)
                        request = predicted_gate[0][-1].data.item()
                        request_index = predicted_gate[0].size(0) - 1
                        if request == 1.:
                            SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, None, None, fp, args=args)
                        else:
                            concat_flight = torch.zeros((1, 1, 256)); concat_flight = concat_flight.cuda()
                            logits_train1, sequence_symbols, teacher_sequence_symbols, predicted_gate = model.module.A_Inference_bleu(source_diag, target_diag, size_dialogue, has_reservation, col_seq, concat_flight, args=args)
                        
                            ################################################################
                            ######################### Prior ################################
                            ################################################################
                            # source and target sentences
                            prior_history = [[] for _ in range(b)]
                            for i in range(b):
                                for j in source_diag[i]:
                                    prior_history[i].append(j.data.item())
                            prior_sents = [[] for _ in range(b)]
                            prior_length = [0 for _ in range(b)]
                            for s in range(len(sequence_symbols)):
                                for i in range(b):
                                    token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                                    if token == '<t1>' and prior_length[i] == 0:
                                        prior_sents[i].append(token)
                                        prior_length[i] = len(prior_sents[i])
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
                            # # print('history_prior_gate : ', history_prior_gate)
                            # # print('predicted_gate : ', predicted_gate)
                            # # prior gate predicted

                            if torch.sum(prior_predicted_gate[i]).data.item() != 0 and request == 0:
                                # print('predicted_gate : ', predicted_gate.size(), ' prior_predicted_gate : ', prior_predicted_gate.size(), ' request :', request, request_index)
                                fix_predict_gate = predicted_gate
                                fix_predict_gate[0][request_index] = prior_predicted_gate[0][request_index+1]
                                predicted_gate = fix_predict_gate

                                request = 1.
                                request_index = predicted_gate[0].size(0) - 1
                                t2_gate_turn_index = request_index
                                SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, None, None, fp, args=args)

                            ################################################################
                            ######################### Prior ################################
                            ################################################################

                    total_step += 1

                # Write ground truth answer
                p_fp.write(pred_gate + ' | ' + SQL_query + ' | ' + pred + '\n'); p_fp.flush()
                sql_fp.write(SQL_query + '\n'); sql_fp.flush()
                gate_fp.write(str(t2_gate_turn_index) + '\n'); gate_fp.flush()

                progress_bar(batch_idx, len(dataloader))

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
        if args.mode == 'bleu_t1t2':
            print('Eval on bleu_t1&t2 !')
            self._test_epoches_t1t2(dataloader, model, step, save_dir=save_dir, args=args)
        else:
            print('Please choose t1 | t2 mode !')
            raise
        return model
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

class SupervisedInference(object):
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

    def _test_epoches_t1(self, dataloader, model, start_step, save_dir='runs/exp', args=None):

        if args.syn:
            data_path = 'results/synthesized/'
        elif args.air:
            data_path = 'results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        model.eval()
        total_step = start_step
        if not os.path.exists(data_path + 'Inference_Bleu/t1/'):
            os.makedirs(data_path + 'Inference_Bleu/t1/')
        t1_fp = open(data_path + 'Inference_Bleu/t1/t1_output.txt', 'w')
        with torch.no_grad():
            train_loss = 0
            total = 0

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda()

                if source_diag[0, -1] == 2: # <t2>
                    continue
                if args.sql:
                    logits_train1, sequence_symbols, teacher_sequence_symbols = model.module.C_Inference_bleu(intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, args=args)
                    
                # ('sequence_symbols : ', sequence_symbols)
                b = source_diag.size(0)
                sents = [[] for _ in range(b)]
                length = [0 for _ in range(b)]
                for s in range(len(sequence_symbols)):
                    for i in range(b):
                        token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                        if token == '<t2>' and length[i] == 0:
                            sents[i].append(token)
                            length[i] = len(sents[i])
                        elif token != '<t2>' and length[i] == 0:
                            sents[i].append(token)
                teacher_sents = [[] for _ in range(b)]
                teacher_length = [0 for _ in range(b)]
                for s in range(len(teacher_sequence_symbols)):
                    for i in range(b):
                        token = self.corpus.dictionary.idx2word[teacher_sequence_symbols[s][i]]
                        if token == '<t2>' and teacher_length[i] == 0:
                            teacher_sents[i].append(token)
                            teacher_length[i] = len(teacher_sents[i])
                        elif token != '<t2>' and teacher_length[i] == 0:
                            teacher_sents[i].append(token)
                # source and target sentences
                source_diag_sents = [[] for _ in range(b)]
                target_diag_sents = [[] for _ in range(b)]
                for i in range(b):
                    for j in source_diag[i]:
                        source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                    for j in target_diag[i]:
                        target_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])  
                # print('*'*100)
                for i in range(b):
                    # print('Input   : ', str(source_diag_sents[i]))
                    # print('Target  : ', str(target_diag_sents[i]))
                    # print('Output  : ', str(sents[i]))
                    # print('Teacher : ', str(teacher_sents[i]))
                    o = ''
                    t = ''
                    for j in sents[i]:
                        o += j + ' '
                    for j in target_diag_sents[i][1:]:
                        t += j + ' '
                    t1_fp.write(o + '\n')
                    t1_fp.write(t + '\n')
                # print('*'*100)
                # if total_step == 8:
                #     raise
                progress_bar(batch_idx, len(dataloader))
                total_step += 1
        t1_fp.close()

    def _test_epoches_t2(self, dataloader, model, start_step, save_dir='runs/exp', args=None):

        if args.syn:
            data_path = 'results/synthesized/'
        elif args.air:
            data_path = 'results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        model.eval()
        total_step = start_step
        if not os.path.exists(data_path + 'Inference_Bleu/t2/'):
            os.makedirs(data_path + 'Inference_Bleu/t2/')
        t2_fp = open(data_path + 'Inference_Bleu/t2/t2_output.txt', 'w')
        with torch.no_grad():
            train_loss = 0
            total = 0

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, kb_true_answer = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda(), kb_true_answer.cuda()

                if source_diag[0, -1] == 1: # <t1>
                    continue
                if args.sql:
                    b = source_diag.size(0)
                    # source and target sentences
                    source_diag_sents = [[] for _ in range(b)]
                    target_diag_sents = [[] for _ in range(b)]
                    for i in range(b):
                        for j in source_diag[i]:
                            source_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                        for j in target_diag[i]:
                            target_diag_sents[i].append(self.corpus.dictionary.idx2word[j.data])
                    # print('*'*100)
                    # for i in range(b):
                    #     print('Input   : ', str(source_diag_sents[i]))
                    #     print('Target  : ', str(target_diag_sents[i]))
                    # print('*'*100)
                    logits_train1, sequence_symbols, teacher_sequence_symbols, predicted_gate = model.module.A_Inference_bleu(intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, col_num, kb_true_answer, args=args)
                    
                # ('sequence_symbols : ', sequence_symbols)
                sents = [[] for _ in range(b)]
                sents_gate = []
                length = [0 for _ in range(b)]
                for s in range(len(sequence_symbols)):
                    for i in range(b):
                        token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                        if token == '<t1>' and length[i] == 0:
                            sents[i].append(token)
                            length[i] = len(sents[i])
                        elif token != '<t1>' and length[i] == 0:
                            sents[i].append(token)
                for h in range(len(source_diag_sents[i])):
                    if h < predicted_gate[i].size(0):
                        sents_gate.append(str(source_diag_sents[i][h]) + '_' + str(predicted_gate[i][h].data.item()))
                    else:
                        sents_gate.append(source_diag_sents[i][h])
                teacher_sents = [[] for _ in range(b)]
                teacher_length = [0 for _ in range(b)]
                for s in range(len(teacher_sequence_symbols)):
                    for i in range(b):
                        token = self.corpus.dictionary.idx2word[teacher_sequence_symbols[s][i]]
                        if token == '<t1>' and teacher_length[i] == 0:
                            teacher_sents[i].append(token)
                            teacher_length[i] = len(teacher_sents[i])
                        elif token != '<t1>' and teacher_length[i] == 0:
                            teacher_sents[i].append(token)  
                print('*'*100)
                for i in range(b):
                    print('Input   : ', str(source_diag_sents[i]))
                    print('Target  : ', str(target_diag_sents[i]))
                    print('Output  : ', str(sents[i]))
                    print('Gate : ', str(sents_gate))
                    # print('Teacher : ', str(teacher_sents[i]))
                    o = ''
                    t = ''
                    for j in sents[i]:
                        o += j + ' '
                    for j in target_diag_sents[i][1:]:
                        t += j + ' '
                    t2_fp.write(o + '\n')
                    t2_fp.write(t + '\n')
                print('*'*100)
                progress_bar(batch_idx, len(dataloader))
                total_step += 1
        t2_fp.close()

    def _test_epoches_t1t2(self, dataloader, model, start_step, save_dir='runs/exp', args=None):

        if args.syn:
            data_path = 'results/synthesized/'
        elif args.air:
            data_path = 'results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        model.eval()
        total_step = start_step
        if not os.path.exists(data_path + 'Inference_Bleu/t1t2/'):
            os.makedirs(data_path + 'Inference_Bleu/t1t2/')
        t_fp = open(data_path + 'Inference_Bleu/t1t2/dev_inference_out.txt', 'w')
        teacher_fp = open(data_path + 'Inference_Bleu/t1t2/dev_inference_out_teacher.txt', 'w')

        with torch.no_grad():
            train_loss = 0
            total = 0

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq, turn_gate, turn_gate_index) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, kb, has_reservation, col_seq = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda()

                request = 0
                concat_flight_embed = torch.zeros((1, 1, 256)); concat_flight_embed = concat_flight_embed.cuda()
                turn_index = -1
                predict_flight_number = 'None'

                if source_diag[0, -1] == 1: # <t1>
                    if args.sql:
                        b = source_diag.size(0)
                        logits_train1, sequence_symbols, teacher_sequence_symbols = model.module.C_Inference_bleu(intent, size_intent, source_diag, target_diag, size_dialogue, col_seq, args=args)
                        
                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t2>' and length[i] == 0:
                                sents[i].append(token)
                                length[i] = len(sents[i])
                            elif token != '<t2>' and length[i] == 0:
                                sents[i].append(token)
                    teacher_sents = [[] for _ in range(b)]
                    teacher_length = [0 for _ in range(b)]
                    for s in range(len(teacher_sequence_symbols)):
                        for i in range(b):
                            token = self.corpus.dictionary.idx2word[teacher_sequence_symbols[s][i]]
                            if token == '<t2>' and teacher_length[i] == 0:
                                teacher_sents[i].append(token)
                                teacher_length[i] = len(teacher_sents[i])
                            elif token != '<t2>' and teacher_length[i] == 0:
                                teacher_sents[i].append(token)
                    for i in range(b):
                        o = ''
                        t = ''
                        for j in sents[i][:-1]:
                            o += j + ' '
                        for j in teacher_sents[i][:-1]:
                            t += j + ' '
                        t_fp.write(o + '\n')
                        t_fp.flush()
                        teacher_fp.write(t + '\n')
                        teacher_fp.flush()
                    total_step += 1

                if source_diag[0, -1] == 2: # <t2>
                    if args.sql:
                        b = source_diag.size(0)
                        if turn_gate[0][0] == 1:
                            request = 1
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
                                # print('kb : ', kb[0].size())
                                global_pointer = model.module.Point_Encode_KB(source_diag, size_dialogue, kb, has_reservation, col_seq, args=args)
                                # print('global_pointer : ', global_pointer)
                                _, global_pointer_index = torch.max(F.softmax(global_pointer, dim=1).data, 1)
                                # print('global_pointer_index : ', global_pointer_index)
                                concat_flight = kb[0, global_pointer_index]
                                # print('concat_flight : ', concat_flight.size(), concat_flight)
                                predict_flight_number = self.corpus.dictionary.idx2word[concat_flight[0, -1]]
                                # print('predict_flight_number : ', predict_flight_number)
                                concat_flight_embed = model.module.Encode_Flight_KB(concat_flight.unsqueeze(0))
                            logits_train1, sequence_symbols, teacher_sequence_symbols, predicted_gate = model.module.A_Inference_bleu(source_diag, target_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                            turn_index = predicted_gate[0].size(0) - 1
                        if request == 1:
                            logits_train1, sequence_symbols, teacher_sequence_symbols, predicted_gate = model.module.A_Inference_bleu(source_diag, target_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, turn_gate=turn_index, args=args)
                        else:
                            logits_train1, sequence_symbols, teacher_sequence_symbols, predicted_gate = model.module.A_Inference_bleu(source_diag, target_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                    
                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t1>' and length[i] == 0:
                                sents[i].append(token)
                                length[i] = len(sents[i])
                            elif token != '<t1>' and length[i] == 0:
                                sents[i].append(token)
                    teacher_sents = [[] for _ in range(b)]
                    teacher_length = [0 for _ in range(b)]
                    for s in range(len(teacher_sequence_symbols)):
                        for i in range(b):
                            token = self.corpus.dictionary.idx2word[teacher_sequence_symbols[s][i]]
                            if token == '<t1>' and teacher_length[i] == 0:
                                teacher_sents[i].append(token)
                                teacher_length[i] = len(teacher_sents[i])
                            elif token != '<t1>' and teacher_length[i] == 0:
                                teacher_sents[i].append(token)  
                    for i in range(b):
                        o = ''
                        t = ''
                        for j in sents[i][:-1]:
                            o += j + ' '
                        for j in teacher_sents[i][:-1]:
                            t += j + ' '
                        t_fp.write(o + '\n')
                        t_fp.flush()
                        teacher_fp.write(t + '\n')
                        teacher_fp.flush()
                    total_step += 1

                progress_bar(batch_idx, len(dataloader))
        t_fp.close()
        teacher_fp.close()

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
        if args.mode == 'bleu_t1':
            print('Eval on bleu_t1 !')
            self._test_epoches_t1(dataloader, model, step, save_dir=save_dir, args=args)
        elif args.mode == 'bleu_t2':
            print('Eval on bleu_t2 !')
            self._test_epoches_t2(dataloader, model, step, save_dir=save_dir, args=args)
        elif args.mode == 'bleu_t1t2':
            print('Eval on bleu_t1&t2 !')
            self._test_epoches_t1t2(dataloader, model, step, save_dir=save_dir, args=args)
        else:
            print('Please choose t1 | t2 mode !')
            raise
        return model
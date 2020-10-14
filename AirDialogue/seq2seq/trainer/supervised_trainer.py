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

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', model_dir='checkpoints/', batch_size=64, random_seed=None, checkpoint_every=100, print_every=100, args=None):
        self._trainer = "Simple Trainer"
        self.loss = loss
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir

        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        self.sql = args.sql
        self.args = args

    def _train_epoches(self, dataloader, model, n_epochs, start_epoch, start_step, dev_data=None, teacher_forcing_ratio=0, clip=5.0, save_dir='runs/exp', args=None):

        writer = SummaryWriter(save_dir)
        model.train()
        total_step = start_step
        str_name = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', 'price', 'num_connections']
        
        for epoch in range(start_epoch, n_epochs + 1):
            print('Epoch : ', epoch)
            train_loss = 0
            total = 0
            correct_firstname, correct_lastname, correct_flight, correct_state, correct_flight2 = 0, 0, 0, 0, 0
            efc = 0
            total_gp_correct = [0, 0]
            total_gp = [0, 0]
            loss1_logits1_mask, loss2_logits2_mask, loss3_logits3_rnn = 0, 0, 0
            loss_firstname, loss_lastname, loss_flight, loss_state = 0, 0, 0, 0 

            correct_num, correct_col, correct_op, correct_str, correct_gate = 0, 0, 0, 0, 0; correct_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            loss_num, loss_col, loss_op, loss_str, loss_gate = 0, 0, 0, 0, 0; loss_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            total_num, total_col, total_op, total_str, total_gate = 0, 0, 0, 0, 0; total_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation, predicted_action, reward_diag, reward_action, mask1, mask2, turn_point, col_seq, col_num, ans_seq, truth_seq, state_tracking_label, gate_label, state_tracking_list, SQL_YN, Info, kb_true_answer_onehot, eval_step, kb_true_answer) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation, predicted_action, reward_diag, reward_action, mask1, mask2, turn_point, col_seq, state_tracking_label, gate_label, SQL_YN, Info, kb_true_answer_onehot, eval_step, kb_true_answer = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), action.cuda(), size_action.cuda(), kb.cuda(), has_reservation.cuda(), predicted_action.cuda(), reward_diag.cuda(), reward_action.cuda(), mask1.cuda(), mask2.cuda(), turn_point.cuda(), col_seq.cuda(), state_tracking_label.cuda(), gate_label.cuda(), SQL_YN.cuda(), Info.cuda(), kb_true_answer_onehot.cuda(), eval_step.cuda(), kb_true_answer.cuda()

                if args.sql:
                    logits_train1, logits_train2, logits_train3, cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, global_pointer = model(source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args)
                    sl_loss, each_loss, correct = compute_loss_nn(logits_train1, logits_train2, logits_train3, target_diag, size_dialogue, predicted_action, size_action, mask1, mask2, kb_true_answer_onehot, args)
                    total_sql_loss, sql_loss, sql_correct, sql_total, sql_str_each = compute_sql_loss(ans_seq, cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, state_tracking_label, SQL_YN, gate_label, state_tracking_list, args=args)
                    gp_loss, gp_correct, gp_total = compute_att_loss(global_pointer, kb_true_answer_onehot, kb_true_answer, args)
                else:
                    logits_train1, logits_train2, logits_train3 = model(source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args)
                    sl_loss, each_loss, correct = compute_loss_nn(logits_train1, logits_train2, logits_train3, target_diag, size_dialogue, predicted_action, size_action, mask1, mask2, kb_true_answer_onehot, args)
                    total_sql_loss, sql_loss, sql_correct, sql_total = 0, 0, 0, 0

                self.optimizer.zero_grad()
                if args.sql:
                    loss = args.sql_alpha1 * total_sql_loss + sl_loss + gp_loss
                else:
                    loss = sl_loss
                loss.backward()

                #  norm_type '1, 2, 'inf'
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, norm_type=2)

                self.optimizer.step()
                train_loss += sl_loss.item()

                total += action.size(0)

                correct_firstname += correct[0].item() ; 
                correct_lastname += correct[1].item() ;
                correct_flight += correct[2].item(); 
                correct_state += correct[3].item() ; 

                total_gp_correct[0] += gp_correct[0]; total_gp[0] += gp_total[0]
                total_gp_correct[1] += gp_correct[1]; total_gp[1] += gp_total[1]

                loss1_logits1_mask += each_loss[0].item() ; 
                loss2_logits2_mask += each_loss[1].item() ; 
                loss3_logits3_rnn += each_loss[2].item() ; 

                if args.sql:
                    correct_num += sql_correct[0] ; 
                    correct_col += sql_correct[1] ; 
                    correct_op  += sql_correct[2] ; 
                    correct_str += sql_correct[3] ; 
                    correct_gate += sql_correct[4];
                    for c in range(11):
                        correct_str_each[c] += sql_str_each[1][c]

                    loss_num += sql_loss[0].item() ; 
                    loss_col += sql_loss[1].item(); 
                    loss_op  += sql_loss[2].item() ; 
                    loss_str += sql_loss[3].item() ; 
                    loss_gate += sql_loss[4].item() ;
                    for c in range(11):
                        loss_str_each[c] += sql_str_each[0][c]

                    total_num += sql_total[0];
                    total_col += sql_total[1] ;
                    total_op  += sql_total[2] ;
                    total_str += sql_total[3] ;
                    total_gate += sql_total[4] ;
                    for c in range(11):
                        total_str_each[c] += sql_str_each[2][c]

                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc : %.1f%% (%d/%d) | %.1f%% (%d/%d) | %.1f%% |: %.1f%% |: %.1f%% |: %.1f%%'
                     % (train_loss / (batch_idx + 1), \
                        100. * total_gp_correct[0] / total_gp[0], total_gp_correct[0], total_gp[0], \
                        100. * total_gp_correct[1] / total_gp[1], total_gp_correct[1], total_gp[1], \
                        100. * correct_firstname / total, \
                        100. * correct_lastname / total, \
                        100. * correct_flight / (total), \
                        100. * correct_state / total))

                writer.add_scalar('train_logit1_loss ', loss1_logits1_mask / (batch_idx + 1), total_step)
                writer.add_scalar('train_logit2_loss ', loss2_logits2_mask / (batch_idx + 1), total_step)
                writer.add_scalar('train_logit3_loss ', loss3_logits3_rnn / (batch_idx + 1), total_step)

                writer.add_scalar('train_acc_first_name', correct_firstname / total, total_step)
                writer.add_scalar('train_acc_last_name', correct_lastname / total, total_step)
                if args.nnkb:
                    writer.add_scalar('train_acc_flight', correct_flight / total, total_step)
                if args.sigmkb:
                    writer.add_scalar('train_acc_flight', 1. * correct_flight / total, total_step)
                    writer.add_scalar('train_gp0', 1. * total_gp_correct[0] / total_gp[0], total_step)
                    writer.add_scalar('train_gp1', 1. * total_gp_correct[1] / total_gp[1], total_step)
                writer.add_scalar('train_acc_state', correct_state / total, total_step)

                if args.sql:
                    writer.add_scalar('train_sql_acc_num', correct_num / total_num, total_step)
                    writer.add_scalar('train_sql_acc_col', correct_col / total_col, total_step)
                    writer.add_scalar('train_sql_acc_op' , correct_op  / total_op , total_step)
                    writer.add_scalar('train_sql_acc_str', correct_str / total_str, total_step)
                    writer.add_scalar('train_sql_acc_gate', correct_gate / total_gate, total_step)
                    writer.add_scalar('train_sql_loss_num', loss_num / (batch_idx + 1), total_step)
                    writer.add_scalar('train_sql_loss_col', loss_col / (batch_idx + 1), total_step)
                    writer.add_scalar('train_sql_loss_op' , loss_op  / (batch_idx + 1) , total_step)
                    writer.add_scalar('train_sql_loss_str', loss_str / (batch_idx + 1), total_step)
                    writer.add_scalar('train_sql_loss_gate', loss_gate / (batch_idx + 1), total_step)
                    for param_group in self.optimizer.param_groups:
                        writer.add_scalar('learning_rate', param_group['lr'], total_step)
                    for c in range(11):    
                        writer.add_scalar(str_name[c] + '_each_loss', loss_str_each[c] / (batch_idx + 1), total_step)
                        writer.add_scalar(str_name[c] + '_each_acc', correct_str_each[c] / total_str_each[c], total_step)
                        
                for param_group in self.optimizer.param_groups:
                    writer.add_scalar('learning_rate', param_group['lr'], total_step)

                if total_step % self.checkpoint_every == 0 and total_step != 0:
                    Checkpoint(model=model, optimizer=self.optimizer, epoch=epoch, step=total_step).save(self.model_dir)
                
                total_step += 1
                self.scheduler.step()

    def _test_epoches(self, dataloader, model, n_epochs, start_epoch, start_step, dev_data=None, teacher_forcing_ratio=0, clip=5.0, save_dir='runs/exp', args=None):

        model.eval()
        total_step = start_step
        str_name = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', 'price', 'num_connections']
        
        if args.syn:
            data_path = 'results/synthesized/'
        elif args.air:
            data_path = 'results/airdialogue/'
        else:
            print('Pleae use --syn or --air !')
            raise

        # eval sql acc
        if args.dev:
            if not os.path.exists(data_path + 'SQL/dev_sql/'):
                os.makedirs(data_path + 'SQL/dev_sql/')
            all_fp = open(data_path + 'SQL/dev_sql/dev_predict_gt', 'w')
            predict_fp = open(data_path + 'SQL/dev_sql/dev_predict_query', 'w')
            gt_fp = open(data_path + 'SQL/dev_sql/dev_gt_query', 'w')
            gate_fp = open(data_path + 'SQL/dev_sql/dev_gate', 'w')
            simple_fp = open(data_path + 'SQL/dev_sql/dev_simple', 'w')
            p_simple_fp = open(data_path + 'SQL/dev_sql/dev_simple_predict', 'w')
            g_simple_fp = open(data_path + 'SQL/dev_sql/dev_simple_ground', 'w')
            fp = [all_fp, predict_fp, gt_fp, gate_fp, simple_fp, p_simple_fp, g_simple_fp]
            total_gate_match = [0, 0]
            total_query_match = [0, 0]
            full_query_match = [0, 0]
        else:
            if not os.path.exists(data_path + 'SQL/train_sql/'):
                os.makedirs(data_path + 'SQL/train_sql/')
            all_fp = open(data_path + 'SQL/train_sql/train_predict_gt', 'w')
            predict_fp = open(data_path + 'SQL/train_sql/train_predict_query', 'w')
            gt_fp = open(data_path + 'SQL/train_sql/train_gt_query', 'w')
            gate_fp = open(data_path + 'SQL/train_sql/train_gate', 'w')
            simple_fp = open(data_path + 'SQL/train_sql/train_simple', 'w')
            fp = [all_fp, predict_fp, gt_fp, gate_fp, simple_fp]
            total_gate_match = [0, 0]
            total_query_match = [0, 0]
            full_query_match = [0, 0]

        with torch.no_grad():
            train_loss = 0
            total = 0
            total_gp_correct = [0, 0]
            total_gp = [0, 0]
            correct_firstname, correct_lastname, correct_flight, correct_state, correct_flight2 = 0, 0, 0, 0, 0
            loss1_logits1_mask, loss2_logits2_mask, loss3_logits3_rnn = 0, 0, 0
            loss_firstname, loss_lastname, loss_flight, loss_state = 0, 0, 0, 0 

            correct_num, correct_col, correct_op, correct_str, correct_gate = 0, 0, 0, 0, 0; correct_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            loss_num, loss_col, loss_op, loss_str, loss_gate = 0, 0, 0, 0, 0; loss_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            total_num, total_col, total_op, total_str, total_gate = 0, 0, 0, 0, 0; total_str_each = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            total_ACC_lf_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            total_truth_gate_num = 0

            for batch_idx, (intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation, predicted_action, reward_diag, reward_action, mask1, mask2, turn_point, col_seq, col_num, ans_seq, truth_seq, state_tracking_label, gate_label, state_tracking_list, SQL_YN, Info, kb_true_answer_onehot, eval_step, kb_true_answer) in enumerate(dataloader):
                
                intent, size_intent, source_diag, target_diag, size_dialogue, action, size_action, kb, has_reservation, predicted_action, reward_diag, reward_action, mask1, mask2, turn_point, col_seq, state_tracking_label, gate_label, SQL_YN, Info, kb_true_answer_onehot, eval_step, kb_true_answer = intent.cuda(), size_intent.cuda(), source_diag.cuda(), target_diag.cuda(), size_dialogue.cuda(), action.cuda(), size_action.cuda(), kb.cuda(), has_reservation.cuda(), predicted_action.cuda(), reward_diag.cuda(), reward_action.cuda(), mask1.cuda(), mask2.cuda(), turn_point.cuda(), col_seq.cuda(), state_tracking_label.cuda(), gate_label.cuda(), SQL_YN.cuda(), Info.cuda(), kb_true_answer_onehot.cuda(), eval_step.cuda(), kb_true_answer.cuda()

                if args.sql:
                    logits_train1, logits_train2, logits_train3, cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, global_pointer = model(source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args)
                    sl_loss, each_loss, correct = compute_loss_nn(logits_train1, logits_train2, logits_train3, target_diag, size_dialogue, predicted_action, size_action, mask1, mask2, kb_true_answer_onehot, args)
                    total_sql_loss, sql_loss, sql_correct, sql_total, sql_str_each = compute_sql_loss(ans_seq, cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, state_tracking_label, SQL_YN, gate_label, state_tracking_list, args=args)
                    gp_loss, gp_correct, gp_total = compute_att_loss(global_pointer, kb_true_answer_onehot, kb_true_answer, args)
                    # eval sql acc
                    # gate_match, query_match, full_match, ACC_lf_correct, truth_gate_num = QUERY_Output(cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, size_dialogue, state_tracking_label, truth_seq, SQL_YN, fp, args=args)
                    gate_match, query_match, full_match, ACC_lf_correct, truth_gate_num = QUERY_Output_predicted(cond_op_score, cond_col_score, cond_num_score, cond_str_out, cond_gate_score, size_dialogue, state_tracking_label, truth_seq, SQL_YN, fp, args=args)
                    total_gate_match[0] += gate_match[0]; total_gate_match[1] += gate_match[1]
                    total_query_match[0] += query_match[0]; total_query_match[1] += query_match[1]
                    full_query_match[0] += full_match[0]; full_query_match[1] += full_match[1]
                    total_truth_gate_num += truth_gate_num
                    for c in range(12):
                        total_ACC_lf_correct[c] += ACC_lf_correct[c]

                else:
                    logits_train1, logits_train2, logits_train3 = model(source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args)
                    sl_loss, each_loss, correct = compute_loss_nn(logits_train1, logits_train2, logits_train3, target_diag, size_dialogue, predicted_action, size_action, mask1, mask2, kb_true_answer_onehot, args)
                    total_sql_loss, sql_loss, sql_correct, sql_total = 0, 0, 0, 0

                if args.sql:
                    loss = args.sql_alpha1 * total_sql_loss + sl_loss
                else:
                    loss = sl_loss
                train_loss += sl_loss

                total += action.size(0)

                correct_firstname += correct[0]; 
                correct_lastname += correct[1];
                if args.nnkb:
                    correct_flight += correct[2];
                if args.sigmkb:
                    correct_flight += correct[2]; 
                correct_state += correct[3]; 

                loss1_logits1_mask += each_loss[0]; 
                loss2_logits2_mask += each_loss[1]; 
                loss3_logits3_rnn += each_loss[2];

                total_gp_correct[0] += gp_correct[0]; total_gp[0] += gp_total[0]
                total_gp_correct[1] += gp_correct[1]; total_gp[1] += gp_total[1]

                if args.sql:
                    correct_num += sql_correct[0] ; 
                    correct_col += sql_correct[1] ; 
                    correct_op  += sql_correct[2] ; 
                    correct_str += sql_correct[3] ; 
                    correct_gate += sql_correct[4];
                    for c in range(11):
                        correct_str_each[c] += sql_str_each[1][c]

                    loss_num += sql_loss[0] ; 
                    loss_col += sql_loss[1]; 
                    loss_op  += sql_loss[2] ; 
                    loss_str += sql_loss[3] ; 
                    loss_gate += sql_loss[4];
                    for c in range(11):
                        loss_str_each[c] += sql_str_each[0][c]

                    total_num += sql_total[0];
                    total_col += sql_total[1] ;
                    total_op  += sql_total[2] ;
                    total_str += sql_total[3] ;
                    total_gate += sql_total[4] ;
                    for c in range(11):
                        total_str_each[c] += sql_str_each[2][c]

                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc : %.1f%% (%d/%d) | %.1f%% (%d/%d) | %.1f%% |: %.1f%% |: %.1f%% |: %.1f%%'
                     % (train_loss / (batch_idx + 1), \
                        100. * total_gp_correct[0] / total_gp[0], total_gp_correct[0], total_gp[0], \
                        100. * total_gp_correct[1] / total_gp[1], total_gp_correct[1], total_gp[1], \
                        100. * correct_firstname / total, \
                        100. * correct_lastname / total, \
                        100. * correct_flight / (total), \
                        100. * correct_state / total))

                total_step += 1

            print('STR : %.1f%% (%d/%d) |COL_NUM : %.1f%% (%d/%d) |COL : %.1f%% (%d/%d) |OP : %.1f%% (%d/%d) |GATE : %.1f%% (%d/%d)'
                            % ( 100. * correct_str / total_str, correct_str, total_str, \
                                100. * correct_num / total_num, correct_num, total_num, \
                                100. * correct_col / total_col, correct_col, total_col, \
                                100. * correct_op / total_op, correct_op, total_op, \
                                100. * correct_gate / total_gate, correct_gate, total_gate))
            print('num', 100. * correct_num / total_num)
            print('col', 100. * correct_col / total_col)
            print('op' , 100. * correct_op  / total_op )
            print('str', 100. * correct_str / total_str)
            print('gate', 100. * correct_gate / total_gate)
            for c in range(11):    
                print(str_name[c], 100. * correct_str_each[c] / total_str_each[c], correct_str_each[c], total_str_each[c])
            print('Gate : ', total_gate_match, ' : ', 100.*total_gate_match[0]/total_gate_match[1])
            print('Query : ', total_query_match, ' : ', 100.*total_query_match[0]/total_query_match[1])
            print('Full Query : ', full_query_match, ' : ', 100.*full_query_match[0]/full_query_match[1])
            for c in range(12):
                print('condiction : ', c, 'ACC_lf_correct : ', 100.*total_ACC_lf_correct[c] / total_truth_gate_num, total_ACC_lf_correct[c], total_truth_gate_num)
            print('End eval')

    def train(self, args, model, dataloader, scheduler, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=1.0, save_dir='runs/exp'):

        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.model_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            # model = resume_checkpoint.model
            model.load_state_dict(resume_checkpoint.model)
            self.optimizer = optimizer
            self.args = args
            model.args = args
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            print('Resume from ', latest_checkpoint_path)
            print('start_epoch : ', start_epoch)
            print('step : ', step)

            if args.adam:
                self.optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(resume_checkpoint.optimizer)
            elif args.sgd:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(resume_checkpoint.optimizer)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, args.decay_steps, gamma=args.decay_factor)
            # for i in range(step):
            #     self.scheduler.step()
            self.scheduler._step_count = step
            for param_group in self.optimizer.param_groups:
                print('learning rate', param_group['lr'], step)
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer
            self.scheduler = scheduler

        # self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))
        if args.only_sql:
            self._train_sql_epoches(dataloader, model, num_epochs, start_epoch, step, dev_data=dev_data, teacher_forcing_ratio=teacher_forcing_ratio, clip=args.clip, save_dir=save_dir, args=args)
        else:
            self._train_epoches(dataloader, model, num_epochs, start_epoch, step, dev_data=dev_data, teacher_forcing_ratio=teacher_forcing_ratio, clip=args.clip, save_dir=save_dir, args=args)
        return model

    def test(self, args, model, dataloader, scheduler, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=1.0, save_dir='runs/exp'):

        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.model_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            # model = resume_checkpoint.model
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
        self._test_epoches(dataloader, model, num_epochs, start_epoch, step, dev_data=dev_data, teacher_forcing_ratio=teacher_forcing_ratio, clip=args.clip, save_dir=save_dir, args=args)
        return model

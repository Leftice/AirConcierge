import os
import argparse
import logging

import torch
from torch.optim import lr_scheduler
import sys

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.trainer import SupervisedInference
from seq2seq.trainer import SupervisedInferencePrior
from seq2seq.trainer import SupervisedSelfPlayEval
from seq2seq.trainer import SupervisedSelfPlayPrior
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.util.checkpoint import Checkpoint

from dataloader import *

import torch.backends.cudnn as cudnn
import random
import torch.nn as nn
import os
import json

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

net_hidden = 256
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment', help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')

#parameter
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')

# intent RNN
parser.add_argument('--intent_bidirectional', action='store_true', help='Using intent_bidirectional RNN')
parser.add_argument('--intent_dropout_p', default=0.2, type=float, help='dropout_p of intent rnn') # dropout works between layers but not every timestep
parser.add_argument('--intent_n_layer', default=1, type=int, help='intent_n_layer')
parser.add_argument('--intent_input_size', default=net_hidden, type=int, help='intent_input_size')
parser.add_argument('--intent_hidden_size', default=net_hidden, type=int, help='intent_hidden_size')

# kb RNN
parser.add_argument('--kb_bidirectional', action='store_true', help='Using kb_bidirectional RNN')
parser.add_argument('--kb_dropout_p', default=0.2, type=float, help='dropout_p of kb rnn')
parser.add_argument('--kb_n_layer', default=1, type=int, help='kb_n_layer')
parser.add_argument('--kb_input_size', default=net_hidden, type=int, help='kb_input_size')
parser.add_argument('--kb_hidden_size', default=net_hidden, type=int, help='kb_hidden_size')

# projection encoder
parser.add_argument('--projection1_input_size', default=net_hidden, type=int, help='projection1_input_size')
parser.add_argument('--projection1_num_units', default=net_hidden, type=int, help='projection1_num_units')
parser.add_argument('--projection2_input_size', default=net_hidden, type=int, help='projection1_input_dim')
parser.add_argument('--projection2_num_units', default=net_hidden, type=int, help='projection1_num_units')

# encoder1 
parser.add_argument('--encoder1_bidirectional', action='store_true', help='Using encoder1_bidirectional RNN')
parser.add_argument('--encoder1_dropout_p', default=0.2, type=float, help='dropout_p of encoder1 rnn')
parser.add_argument('--encoder1_n_layer', default=2, type=int, help='encoder1_n_layer')
parser.add_argument('--encoder1_input_size', default=net_hidden, type=int, help='encoder1_input_size')
parser.add_argument('--encoder1_hidden_size', default=net_hidden, type=int, help='encoder1_hidden_size')

# encoder2
parser.add_argument('--encoder2_bidirectional', action='store_true', help='Using encoder2_bidirectional RNN')
parser.add_argument('--encoder2_dropout_p', default=0.2, type=float, help='dropout_p of encoder2 rnn')
parser.add_argument('--encoder2_n_layer', default=2, type=int, help='encoder2_n_layer')
parser.add_argument('--encoder2_input_size', default=net_hidden, type=int, help='encoder2_input_size')
parser.add_argument('--encoder2_hidden_size', default=net_hidden, type=int, help='encoder2_hidden_size')

# decoder1
parser.add_argument('--decoder1_bidirectional', action='store_true', help='Using decoder1_bidirectional RNN')
parser.add_argument('--decoder1_dropout_p', default=0.2, type=float, help='dropout_p of decoder1 rnn')
parser.add_argument('--decoder1_n_layer', default=2, type=int, help='decoder1_n_layer')
parser.add_argument('--decoder1_input_size', default=net_hidden, type=int, help='decoder1_input_size')
parser.add_argument('--decoder1_hidden_size', default=net_hidden, type=int, help='decoder1_hidden_size')

# decoder2
parser.add_argument('--decoder2_bidirectional', action='store_true', help='Using decoder2_bidirectional RNN')
parser.add_argument('--decoder2_dropout_p', default=0.2, type=float, help='dropout_p of decoder2 rnn')
parser.add_argument('--decoder2_n_layer', default=2, type=int, help='decoder2_n_layer')
parser.add_argument('--decoder2_input_size', default=net_hidden, type=int, help='decoder2_input_size')
parser.add_argument('--decoder2_hidden_size', default=net_hidden, type=int, help='decoder2_hidden_size')

# decoder3
parser.add_argument('--decoder3_bidirectional', action='store_true', help='Using decoder3_bidirectional RNN')
parser.add_argument('--decoder3_dropout_p', default=0.2, type=float, help='dropout_p of decoder3 rnn')
parser.add_argument('--decoder3_n_layer', default=1, type=int, help='decoder3_n_layer')
parser.add_argument('--decoder3_input_size', default=net_hidden, type=int, help='decoder3_input_size')
parser.add_argument('--decoder3_hidden_size', default=net_hidden, type=int, help='decoder3_hidden_size')

# decoder output layer 1 + layer2 + action
parser.add_argument('--output_layer1_input_size', default=net_hidden, type=int, help='output_layer1_input_size')
parser.add_argument('--output_layer2_input_size', default=net_hidden, type=int, help='output_layer2_input_size')
parser.add_argument('--output_layer3_input_size', default=net_hidden, type=int, help='output_layer3_input_size')

# action
parser.add_argument('--action_name_hidden', default=net_hidden, type=int, help='action_name_hidden')
parser.add_argument('--action_flight_hidden', default=net_hidden, type=int, help='action_name_hidden')
parser.add_argument('--action_state_hidden', default=net_hidden, type=int, help='action_name_hidden')

parser.add_argument('--vocab_size', default=5329, type=int, help='vocab_size')
parser.add_argument('--embedding_encoder_hidden_size', default=net_hidden, type=int, help='embedding_encoder_hidden_size')

# wei wei 
parser.add_argument('--init_weight', type=float, default=0.1, help='for uniform init_op, initialize weights " "between [-this, this].')
parser.add_argument('--max_len', type=int, default=500, help='maximum length for a dialogue during training')
parser.add_argument('--max_inference_len', type=int, default=50, help='maximum sentence length for dialogue inference')
parser.add_argument('--max_dialogue_turns', type=int, default=50, help='The maximum number of turns that a dialogue would take to terminal. When conducting self-play, this is the maximum number of turns we expect the dialogue to reach an end-of-dialogue token.')
parser.add_argument('--num_kb_fields_per_entry', type=int, default=13, help='number of attributes of each flight in the knowledge base')
parser.add_argument('--len_action', type=int, default=4, help='number of dialogue states for each conversation')
parser.add_argument('--start_decay_step', type=int, default=0, help='When we start to decay')
parser.add_argument('--decay_steps', type=int, default=20000, help='How frequent we decay')
parser.add_argument('--decay_factor', type=float, default=0.1, help='How much we decay.')
parser.add_argument('--num_train_steps', type=int, default=12000, help='Num steps to train.')

parser.add_argument('--save_dir', default='runs/exp', type=str, help='save_dir')
parser.add_argument('--model_dir', default='checkpoints/', type=str, help='model_dir')
parser.add_argument('--print_every', default=100, type=int, help='print_every')
parser.add_argument('--checkpoint_every', default=100, type=int, help='checkpoint_every')
parser.add_argument('--action_att', action='store_true', help='Use action attention')

# seq2seq attention
parser.add_argument('--att_method', default='concat', type=str, help='seq2seq attention')
parser.add_argument('--att_type', default='Bahd', type=str, help='seq2seq attention')
parser.add_argument('--att_mlp', action='store_true', help='Use seq2seq attention mlp')

# SQLNet
parser.add_argument('--sql', action='store_true', help='Use toy')
parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet(requires pretrained model).')
parser.add_argument('--use_att', action='store_true', help='Use conditional attention.')
parser.add_argument('--sql_alpha1', type=float, default=1.0, help='learning rate')
parser.add_argument('--sql_alpha2', type=float, default=10.0, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='learning rate')
parser.add_argument('--N_h', type=int, default=100, help='N_h')
parser.add_argument('--acc_print', type=int, default=2, help='N_h')
parser.add_argument('--seed', default=11111, type=int, help='rng seed')

# dataset
parser.add_argument('--n_sample', type=int, default=-1, help='N_h')
parser.add_argument('--smallkb_n', default=30, type=int, help='rng seed')
parser.add_argument('--toy', action='store_true', help='Use toy')
parser.add_argument('--dev', action='store_true', help='Use dev')
parser.add_argument('--eval', action='store_true', help='Use eval')
parser.add_argument('--infer_bleu_prior', action='store_true')
parser.add_argument('--infer_bleu', action='store_true')
parser.add_argument('--self_play_eval_prior', action='store_true')
parser.add_argument('--self_play_eval', action='store_true')
parser.add_argument('--mode', type=str)
parser.add_argument('--only_f', action='store_true')
parser.add_argument('--nnkb', action='store_true')
parser.add_argument('--sigmkb', action='store_true')
parser.add_argument('--sigmkb_alpha', type=float, default=60.0, help='learning rate')
parser.add_argument('--mask', action='store_true', help='Use toy')
parser.add_argument('--shuffle', action='store_true', help='Use shuffle dataset')
parser.add_argument('--adam', action='store_true', help='Use adam')
parser.add_argument('--sgd', action='store_true', help='Use sgd')
parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
parser.add_argument('--eval_step', action='store_true')
parser.add_argument('--only_sql', action='store_true')
parser.add_argument('--init', type=str, default='mos')
parser.add_argument('--syn', action='store_true', help='Use syn')
parser.add_argument('--air', action='store_true', help='Use air')

# parameter setting
args = parser.parse_args()

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# torch.backends.cudnn.deterministic=True
# # If you don't need to reproduce 100% the same results of a certain seed, 
# # the below one can be set to True and it might be slightly faster.
# torch.backends.cudnn.benchmark = True

print('Loading Data ... ')
if args.infer_bleu_prior:
    print('Inference bleu prior!')
    dataloader, corpus = Inference_loader(args.batch_size, args.toy, args.max_len, args.shuffle, args.mask, args.only_f, args.dev, args.n_sample, args.smallkb_n, args)
elif args.infer_bleu:
    print('Inference bleu!')
    dataloader, corpus = Inference_loader_2(args.batch_size, args.toy, args.max_len, args.shuffle, args.mask, args.only_f, args.dev, args.n_sample, args.smallkb_n, args)
elif args.self_play_eval_prior:
    print('Self play prior !')
    dataloader, corpus = SelfPlayEval_loader(args.batch_size, args.toy, args.max_len, args.shuffle, args.mask, args.only_f, args.dev, args.n_sample, args.smallkb_n, args)
elif args.self_play_eval:
    print('Self play eval !')
    dataloader, corpus = SelfPlayEval_loader_2(args.batch_size, args.toy, args.max_len, args.shuffle, args.mask, args.only_f, args.dev, args.n_sample, args.smallkb_n, args)
else:
    print('Supervised dataloader !')
    dataloader, corpus = loader(args.batch_size, args.toy, args.max_len, args.shuffle, args.mask, args.only_f, args.dev, args.n_sample, args.smallkb_n, args)

seq2seq = None
optimizer = None
# Initialize model
hidden_size = net_hidden
bidirectional = False
max_len = 300

args.vocab_size = len(corpus.dictionary.word2idx)

eos_id= corpus.dictionary.word2idx['<eod>']
sos_id = 1

print('Building Model ... ')

decoder1 = DecoderRNN(args.decoder1_n_layer, args.vocab_size, args.max_inference_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.2, use_attention=args.use_att, bidirectional=bidirectional,
                     eos_id=sos_id+1, sos_id=sos_id, batch_size=args.batch_size, att_method=args.att_method, att_mlp=args.att_mlp, att_type=args.att_type)
decoder2 = DecoderRNN(args.decoder2_n_layer, args.vocab_size, args.max_inference_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.2, use_attention=args.use_att, bidirectional=bidirectional,
                     eos_id=sos_id, sos_id=sos_id+1, batch_size=args.batch_size, att_method=args.att_method, att_mlp=args.att_mlp, att_type=args.att_type)
# decoder3 = DecoderRNN(args.decoder3_n_layer, args.vocab_size, max_len, hidden_size * 2 if bidirectional else hidden_size,
#                      dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
#                      eos_id=eos_id, sos_id=sos_id)

# seq2seq = Seq2seq(args, decoder1, decoder2, decoder3)
seq2seq = Seq2seq(args, decoder1, decoder2)
seq2seq.cuda()
seq2seq = torch.nn.DataParallel(seq2seq)
cudnn.benchmark = True

print('Initialize model parameter ...')
if args.init == 'uniform':
    print('uniform init !')
    for param in seq2seq.parameters():
        param.data.uniform_(-args.init_weight, args.init_weight)
elif args.init == 'mos':
    print('mos init !')
    for m in seq2seq.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    # torch.nn.init.xavier_uniform_(param.data)
                    torch.nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'weight_hh' in name:
                    # torch.nn.init.orthogonal_(param.data)
                    torch.nn.init.uniform_(param.data, -0.1, 0.1)
                elif 'bias' in name:
                    param.data.fill_(0)
else:
    print('default init !')

total_parameter = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
# print('Total parameter : ', total_parameter)

# optimizer
if args.adam:
    print('Using ADAM !')
    optimizer = torch.optim.Adam(seq2seq.parameters())
elif args.sgd:
    print('Using SGD ! LR DF DS: ', args.lr, args.decay_factor, args.decay_steps)
    optimizer = torch.optim.SGD(seq2seq.parameters(), lr=args.lr)
else:
    print('No optimizer !')
    raise
scheduler = lr_scheduler.StepLR(optimizer, args.decay_steps, gamma=args.decay_factor)

# option
if args.infer_bleu:
    print('Inference bleu ... ')
    t = SupervisedInference(model_dir=args.model_dir, args=args, corpus=corpus)
    seq2seq = t.test(args, seq2seq, dataloader, resume=args.resume, save_dir=args.save_dir)
elif args.infer_bleu_prior:
    print('Inference bleu Prior ... ')
    t = SupervisedInferencePrior(model_dir=args.model_dir, args=args, corpus=corpus)
    seq2seq = t.test(args, seq2seq, dataloader, resume=args.resume, save_dir=args.save_dir)
elif args.self_play_eval_prior:
    print('Self play Prior ... ')
    # t = SupervisedSelfPlayEval(model_dir=args.model_dir, args=args, corpus=corpus)
    t = SupervisedSelfPlayPrior(model_dir=args.model_dir, args=args, corpus=corpus)
    seq2seq = t.test(args, seq2seq, dataloader, resume=args.resume, save_dir=args.save_dir)
elif args.self_play_eval:
    print('Self play eval ... ')
    t = SupervisedSelfPlayEval(model_dir=args.model_dir, args=args, corpus=corpus)
    seq2seq = t.test(args, seq2seq, dataloader, resume=args.resume, save_dir=args.save_dir)
else:
    print('SupervisedTrainer ... ')
    # train
    t = SupervisedTrainer(batch_size=args.batch_size, checkpoint_every=args.checkpoint_every, print_every=args.print_every, expt_dir=args.expt_dir, model_dir=args.model_dir, args=args)
    if args.eval:
        seq2seq = t.test(args, seq2seq, dataloader, scheduler, num_epochs=5, dev_data=None, optimizer=optimizer, teacher_forcing_ratio=1.0, resume=args.resume, save_dir=args.save_dir)
    else:
        seq2seq = t.train(args, seq2seq, dataloader, scheduler, num_epochs=5, dev_data=None, optimizer=optimizer, teacher_forcing_ratio=1.0, resume=args.resume, save_dir=args.save_dir)


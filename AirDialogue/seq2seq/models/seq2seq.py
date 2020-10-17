import torch.nn as nn
import torch.nn.functional as F
from .baseRNN import BaseRNN
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.autograd import Variable
import math
from .MemN2N import ExternalKnowledge

class Seq2seq(nn.Module):

    def __init__(self, args, decoder1, decoder2, decoder3=None, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.args = args
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        # self.decoder3 = decoder3
        self.decode_function = decode_function
        self.loss_action =  nn.CrossEntropyLoss()
        self.loss =  nn.CrossEntropyLoss(reduction='none')

        # init word encoder & decoder embedding
        self.embedding_encoder =  nn.Embedding(args.vocab_size, args.embedding_encoder_hidden_size)
        self.embedding_encoder.weight.requires_grad = True
        self.embedding_decoder = self.embedding_encoder
        
        self.decoder1.embedding = self.embedding_decoder
        self.decoder2.embedding = self.embedding_decoder

        # init intent encoder RNN
        self.intent_encoder_rnn = nn.GRU(args.intent_input_size, args.intent_hidden_size, args.intent_n_layer, batch_first=True, bidirectional=args.intent_bidirectional)

        # init kb encoder hierarchial RNN
        # hierarchial_rnn_1
        self.kb0_encoder_rnn = nn.GRU(args.kb_input_size, args.kb_hidden_size, 2, batch_first=True, bidirectional=True, dropout=args.kb_dropout_p)
        # hierarchial_rnn_2
        self.kb1_encoder_rnn = nn.GRU(args.kb_input_size, args.kb_hidden_size, 2, batch_first=True, bidirectional=True, dropout=args.kb_dropout_p)

        # encoder1 projection1
        projection1_input_size = args.intent_hidden_size + args.embedding_encoder_hidden_size
        projection2_input_size = args.kb_hidden_size + args.embedding_encoder_hidden_size
        self.encoder_input_projection1 = nn.Linear(projection1_input_size, args.projection1_num_units, bias=False)
        self.encoder_input_projection2 = nn.Linear(2*args.embedding_encoder_hidden_size, args.projection2_num_units, bias=False)
        self.sql_encoder_input_projection2 = nn.Linear(args.embedding_encoder_hidden_size, args.projection2_num_units, bias=False)
        self.sql_encoder_input_projection2_2 = nn.Linear(args.embedding_encoder_hidden_size, args.projection2_num_units, bias=False)

        # _build_encoder
        self._build_encoder1 = nn.GRU(args.projection1_num_units, args.encoder1_hidden_size, args.encoder1_n_layer, batch_first=True, bidirectional=args.encoder1_bidirectional, dropout=args.encoder1_dropout_p)
        self._build_encoder2 = nn.GRU(args.projection2_num_units, args.encoder2_hidden_size, args.encoder2_n_layer, batch_first=True, bidirectional=args.encoder2_bidirectional, dropout=args.encoder2_dropout_p)
        
        # logits decoder1, logits deocder2, deocder action
        self._build_decoder1 = nn.GRU(args.decoder1_input_size, args.decoder1_hidden_size, args.decoder1_n_layer, batch_first=True, bidirectional=args.decoder1_bidirectional, dropout=args.decoder1_dropout_p)
        self._build_decoder2 = nn.GRU(args.decoder2_input_size, args.decoder2_hidden_size, args.decoder2_n_layer, batch_first=True, bidirectional=args.decoder2_bidirectional, dropout=args.decoder2_dropout_p)
        self.decoder1.rnn = self._build_decoder1
        self.decoder2.rnn = self._build_decoder2

        # decoder output 1
        self.output_layer1 = nn.Linear(args.output_layer1_input_size, args.vocab_size, bias=False)
        self.decoder1.out = self.output_layer1
        # decoder output 2
        self.output_layer2 = nn.Linear(args.output_layer2_input_size, args.vocab_size, bias=False)
        self.decoder2.out = self.output_layer2

        # decoder output action
        self.v = nn.Parameter(torch.rand(args.action_name_hidden))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        if args.action_att:
            hidd = args.action_name_hidden*2
        else:
            hidd = args.action_name_hidden
        self.decoder3_attention_linear1 = nn.Linear(args.action_name_hidden*2, args.action_name_hidden)
        self.output_layer_action_name1 = nn.Linear(args.action_name_hidden, args.vocab_size, bias=False)
        self.output_layer_action_name_hidden1 = nn.Linear(hidd, args.action_name_hidden, bias=False)
        
        self.decoder3_attention_linear2 = nn.Linear(args.action_name_hidden*2, args.action_name_hidden)
        self.output_layer_action_name2 = nn.Linear(args.action_name_hidden, args.vocab_size, bias=False)
        self.output_layer_action_name_hidden2 = nn.Linear(hidd, args.action_name_hidden, bias=False)

        self.decoder3_attention_linear3 = nn.Linear(args.action_flight_hidden*2, args.action_flight_hidden)
        self.output_layer_action_flight = nn.Linear(args.action_flight_hidden, args.vocab_size, bias=False)
        self.output_layer_action_flight_hidden = nn.Linear(hidd, args.action_flight_hidden, bias=False)

        self.decoder3_attention_linear4 = nn.Linear(args.action_state_hidden*2, args.action_state_hidden)
        self.output_layer_action_state = nn.Linear(args.action_state_hidden, args.vocab_size, bias=False)
        self.output_layer_action_state_hidden = nn.Linear(hidd, args.action_state_hidden, bias=False)

        # SQLNet
        if args.sql:
            N_word = args.embedding_encoder_hidden_size
            N_h = args.N_h
            N_depth = 2
            self.total_column_K = 12
            # score num
            self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_num_col_att = nn.Linear(N_h, 1)
            self.cond_num_col2hid1 = nn.Linear(N_h, args.encoder2_hidden_size)
            self.softmax_dim0 = nn.Softmax(dim=0)
            self.softmax_dim1 = nn.Softmax(dim=1)
            self.softmax_dim2 = nn.Softmax(dim=2)
            self.softmax_dim3 = nn.Softmax(dim=3)
            self.cond_num_att = nn.Linear(args.encoder2_hidden_size, 1)
            self.cond_num_out = nn.Sequential(nn.Linear(args.encoder2_hidden_size, args.encoder2_hidden_size), nn.Tanh(), nn.Linear(args.encoder2_hidden_size, self.total_column_K+1))
            # predict column 
            self.cond_col_att = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_col_out_K = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_col_out_col = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 1))
            # predict op
            self.cond_op_att = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_op_out_K = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_op_out_col = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_op_out = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 2)) # <, <=, None
            # predict string
            self.cond_str_att = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_str_out_K = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_str_out_col = nn.Linear(N_h, args.encoder2_hidden_size)
            # predict string2
            self.cond_str_att_2 = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_str_name_enc_2 = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_str_out_K_2 = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_str_out_col_2 = nn.Linear(N_h, args.encoder2_hidden_size)
            # predict string3
            self.cond_str_att_3 = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_str_name_enc_3 = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_str_out_K_3 = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_str_out_col_3 = nn.Linear(N_h, args.encoder2_hidden_size)
            # predict gate
            self.cond_gate_att = nn.Linear(args.encoder2_hidden_size, N_h)
            self.cond_gate_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2, num_layers=N_depth, batch_first=True, dropout=0.3, bidirectional=True)
            self.cond_gate_out_K = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_gate_out_col = nn.Linear(N_h, args.encoder2_hidden_size)
            self.cond_gate_out = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 1))
            # each column string
            self.cond_str_out_departure_airport = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 50))
            self.cond_str_out_return_airport = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 50))
            self.cond_str_out_departure_month = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 12))
            self.cond_str_out_return_month = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 12))
            self.cond_str_out_departure_day = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 31))
            self.cond_str_out_return_day = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 31))
            self.cond_str_out_departure_time_num = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 3))
            self.cond_str_out_return_time_num = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 3))
            self.cond_str_out_class = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 2))
            self.cond_str_out_price = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 4))
            self.cond_str_out_num_connections = nn.Sequential(nn.ReLU(), nn.Linear(args.encoder2_hidden_size, 2))

        self.extKnow = ExternalKnowledge(vocab=args.vocab_size, embedding_dim=256, dropout=0.1, hop=3, args=args)
        self.proj_cat_kb = nn.Linear(512, 256)

    def flatten_parameters(self):
        self.intent_encoder_rnn.flatten_parameters()
        self.kb0_encoder_rnn.flatten_parameters()
        self.kb1_encoder_rnn.flatten_parameters()
        self._build_encoder1.flatten_parameters()
        self._build_encoder2.flatten_parameters()
        self._build_decoder1.flatten_parameters()
        self._build_decoder2.flatten_parameters()

    def sql_flatten_parameters(self):
        self.cond_num_name_enc.flatten_parameters()
        self.cond_col_name_enc.flatten_parameters()
        self.cond_op_name_enc.flatten_parameters()
        self.cond_str_name_enc.flatten_parameters()
        self.cond_str_name_enc_2.flatten_parameters()
        self.cond_str_name_enc_3.flatten_parameters()
        self.cond_gate_name_enc.flatten_parameters()

    def forward(self, source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)    
        self.flatten_parameters()

        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)
        has_reservation_emb = self.embedding_encoder(has_reservation)

        ################################################################
        ######################### Encoder1 #############################
        ################################################################
        # encoder1_intent
        intent_encoder_emb_inp = self.embedding_encoder(intent)
        intent_output, intent_encoder_state1_aux = self.intent_encoder_rnn(intent_encoder_emb_inp)
        intent_encoder_state1_aux = intent_encoder_state1_aux.squeeze(0).unsqueeze(1)
        intent_encoder_state1_aux_expand = intent_encoder_state1_aux.expand(-1, max_seq_len, -1)
        concat1 = torch.cat((encoder_emb_inp, intent_encoder_state1_aux_expand), 2)
        concat1_embedded = self.encoder_input_projection1(concat1)
        concat1_padded_embedded = nn.utils.rnn.pack_padded_sequence(concat1_embedded, size_dialogue, batch_first=True)
        encoder_outputs1, encoder_state1 = self._build_encoder1(concat1_padded_embedded)
        encoder_outputs1, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs1, batch_first=True)
        encoder_outputs1 = encoder_outputs1 + concat1_embedded

        # decoder 1
        _, logits_train1 = self.decoder1(inputs=source_diag,
                              encoder_hidden=encoder_state1,
                              encoder_outputs=encoder_outputs1,
                              turn_point=turn_point,
                              function=self.decode_function,
                              teacher_forcing_ratio=1.0)

        ################################################################
        ######################### SQL Generator ########################
        ################################################################
        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_num_col, _ = self.cond_num_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_col, _ = self.cond_col_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_op, _ = self.cond_op_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str, _ = self.cond_str_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str_2, _ = self.cond_str_name_enc_2(col_inp_var) # (b, 12, N_h)
            e_cond_str_3, _ = self.cond_str_name_enc_3(col_inp_var) # (b, 12, N_h)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2
            sql_concat2_embedded_2 = self.sql_encoder_input_projection2_2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded_2 = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded_2, size_dialogue, batch_first=True)
            sql_encoder_outputs2_2, sql_encoder_state2_2 = self._build_encoder2(sql_encoder_emb_inp_padded_2)
            sql_encoder_outputs2_2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2_2, batch_first=True)
            encoder_outputs2_without_kb_2 = sql_encoder_outputs2_2

            # SQLNet
            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze() # (b, s, 1)

            # column num
            h_num_enc = encoder_outputs2_without_kb # (b, s, 256)
            cond_num_score = self.cond_num_out(h_num_enc)

            # Predict the column
            e_cond_col = e_cond_col.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_col_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_col = self.cond_col_att(h_col_enc).expand(-1, -1, c, -1) + e_cond_col # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) + self.cond_col_out_col(e_cond_col)).squeeze() # (b, s, c, 1)
            
            # Predict the operator
            e_cond_op = e_cond_op.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_op_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_op = self.cond_op_att(h_op_enc).expand(-1, -1, c, -1) + e_cond_op # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) + self.cond_op_out_col(e_cond_op)).squeeze() # (b, s, c, 3)

            # Predict the condiction string
            e_cond_str = e_cond_str.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str = self.cond_str_att(h_str_enc).expand(-1, -1, c, -1) + e_cond_str # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc = self.cond_str_out_K(K_cond_str) + self.cond_str_out_col(e_cond_str)
            # Predict the condiction string
            e_cond_str_2 = e_cond_str_2.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_2 = encoder_outputs2_without_kb_2.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_2 = self.cond_str_att_2(h_str_enc_2).expand(-1, -1, c, -1) + e_cond_str_2 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_2 = self.cond_str_out_K_2(K_cond_str_2) + self.cond_str_out_col_2(e_cond_str_2)
            # Predict the condiction string
            e_cond_str_3 = e_cond_str_3.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_3 = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_3 = self.cond_str_att_3(h_str_enc_3).expand(-1, -1, c, -1) + e_cond_str_3 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_3 = self.cond_str_out_K_3(K_cond_str_3) + self.cond_str_out_col_3(e_cond_str_3)

            cond_str_out_departure_airport_nn = self.cond_str_out_departure_airport(cond_str_enc[:, :, 0, :]).squeeze() # (b, s, 50)
            cond_str_out_return_airport_nn = self.cond_str_out_return_airport(cond_str_enc[:, :, 1, :]).squeeze() # (b, s, 50)

            cond_str_out_departure_month_nn = self.cond_str_out_departure_month(cond_str_enc_2[:, :, 2, :]).squeeze() # (b, s, 31)
            cond_str_out_return_month_nn = self.cond_str_out_return_month (cond_str_enc_2[:, :, 3, :]).squeeze() # (b, s, 31)
            cond_str_out_departure_day_nn = self.cond_str_out_departure_day(cond_str_enc_2[:, :, 4, :]).squeeze()
            cond_str_out_return_day_nn = self.cond_str_out_return_day(cond_str_enc_2[:, :, 5, :]).squeeze()

            cond_str_out_departure_time_num_nn = self.cond_str_out_departure_time_num(cond_str_enc_3[:, :, 6, :]).squeeze()
            cond_str_out_return_time_num_nn = self.cond_str_out_return_time_num(cond_str_enc_3[:, :, 7, :]).squeeze()
            cond_str_out_class_nn = self.cond_str_out_class(cond_str_enc_3[:, :, 8, :]).squeeze()
            cond_str_out_price_nn = self.cond_str_out_price(cond_str_enc_3[:, :, 9, :]).squeeze()
            cond_str_out_num_connections_nn = self.cond_str_out_num_connections(cond_str_enc_3[:, :, 10, :]).squeeze() 

        ################################################################
        ###################### Encoder2 & Decoder2 #####################
        ################################################################

        # encoder2_init
        has_reservation_emb = has_reservation_emb.unsqueeze(1).expand(-1, max_seq_len, -1)
        concat2 = torch.cat((encoder_emb_inp, has_reservation_emb), 2) # ('kb_encoder_state2_aux_expand : ', (4, 180, 125))
        concat2_embedded = self.encoder_input_projection2(concat2)
        encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(concat2_embedded, size_dialogue, batch_first=True)
        encoder_outputs2, encoder_state2 = self._build_encoder2(encoder_emb_inp_padded)
        encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs2, batch_first=True)
        encoder_outputs2_concat = encoder_outputs2

        # encoder2_kb
        kb_squeeze = kb.view(-1, 13) # [batch_size*30, 13] <--- [batch_size, 30, 13]
        kb_encoder_emb_inp = self.embedding_encoder(kb_squeeze) # [batch_size*30, 13] ---> [batch_size*30, 13, embedding_encoder_size]
        kb_output0, kb_encoder_state1_aux = self.kb0_encoder_rnn(kb_encoder_emb_inp[:, :-1]) # (b*30, 13, 512) (4, b*30, 256)
        kb_encoder_state1_aux = kb_encoder_state1_aux[-1].squeeze(0)
        kb_encoder_state1_aux = kb_encoder_state1_aux.view(kb.size(0), -1, self.args.kb_hidden_size) # (b, 30, v)
        kb_output0_2, kb_encoder_state1_aux_2 = self.kb0_encoder_rnn(kb_encoder_emb_inp) # (b*30, 13, 512) (4, b*30, 256)
        kb_encoder_state1_aux_2 = kb_encoder_state1_aux_2[-1].squeeze(0)
        kb_encoder_state1_aux_2 = kb_encoder_state1_aux_2.view(kb.size(0), -1, self.args.kb_hidden_size) # (b, 30, v)      

        # eval step action
        embedded_eval = None
        for b in range(batch_size):
            if embedded_eval is None:
                embedded_eval = encoder_outputs2_concat[b, state_tracking_list[b][0]].unsqueeze(0).unsqueeze(1) # (b, s, 256) (256,) (1, 1, 256)
            else:
                embedded_eval = torch.cat((embedded_eval, encoder_outputs2_concat[b, state_tracking_list[b][0]].unsqueeze(0).unsqueeze(1)), dim=0)
        global_pointer, _ = self.extKnow.Enough_step_load_memory(kb[:, :, 6:-1], embedded_eval)

        # decoder 2
        kb_true = None
        for b in range(batch_size):
            if kb_true is None:
                if kb_true_answer[b] == 30:
                    kb_true = torch.zeros_like(kb_encoder_state1_aux_2[b, kb_true_answer[b]].unsqueeze(0).unsqueeze(1))
                else:
                    kb_true = kb_encoder_state1_aux_2[b, kb_true_answer[b]].unsqueeze(0).unsqueeze(1)
            else:
                if kb_true_answer[b] == 30:
                    zero_kb_true = torch.zeros_like(kb_encoder_state1_aux_2[b, kb_true_answer[b]].unsqueeze(0).unsqueeze(1))
                    kb_true = torch.cat((kb_true, zero_kb_true), dim=0)
                else:
                    kb_true = torch.cat((kb_true, kb_encoder_state1_aux_2[b, kb_true_answer[b]].unsqueeze(0).unsqueeze(1)), dim=0)
        kb_true_cat = kb_true.expand(-1, max_seq_len, -1) # (b, s, v)
        gate_mask = gate_label.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        kb_true_cat_gate = torch.mul(kb_true_cat, gate_mask)
        encoder_outputs2_concat_kb = torch.cat((encoder_outputs2_concat, kb_true_cat_gate), dim=2)
        encoder_outputs2_concat_kb = self.proj_cat_kb(encoder_outputs2_concat_kb)
        _, logits_train2 = self.decoder2(inputs=source_diag,
                              encoder_hidden=encoder_state2,
                              encoder_outputs=encoder_outputs2_concat_kb,
                              turn_point=turn_point,
                              function=self.decode_function,
                              teacher_forcing_ratio=1.0)

        ################################################################
        ######################### Final Status #########################
        ################################################################
        if args.action_att:

            timestep = encoder_outputs2_concat.size(1)

            encoder_outputs2_concat_action = encoder_outputs2_concat
            embedded = encoder_state2[-1].unsqueeze(0).transpose(0, 1) # (2, b, 256) -> (1, b, 256) -> (b, 1, 256) 
            v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [b, 1, 256]
            hidden = embedded
            hidden = hidden.expand(-1, timestep, -1)

            energy1 = F.relu(self.decoder3_attention_linear1(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
            # energy2 = F.relu(self.decoder3_attention_linear2(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
            # energy3 = F.relu(self.decoder3_attention_linear3(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
            # energy4 = F.relu(self.decoder3_attention_linear4(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
            
            action_mask = torch.arange(timestep)[None, :].cuda() < size_dialogue[:, None] ; action_mask = action_mask.cuda()
            action_mask = ~action_mask
            attn_weights1 = F.softmax(torch.bmm(v, energy1).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
            # attn_weights2 = F.softmax(torch.bmm(v, energy2).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
            # attn_weights3 = F.softmax(torch.bmm(v, energy3).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
            # attn_weights4 = F.softmax(torch.bmm(v, energy4).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
            context1 = attn_weights1.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
            # context2 = attn_weights2.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
            # context3 = attn_weights3.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
            # context4 = attn_weights4.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
            decoder3_input1 = torch.cat([embedded, context1], 2).squeeze(1)
            decoder3_input2 = torch.cat([embedded, context1], 2).squeeze(1)
            decoder3_input3 = torch.cat([embedded, context1], 2).squeeze(1) # [b, 512]
            decoder3_input4 = torch.cat([embedded, context1], 2).squeeze(1)
            
        else:
            encoder_state2_decoder3 = encoder_state2[-1].squeeze(0)
            encoder_state2_decoder3 = encoder_state2_decoder3.view(encoder_state2_decoder3.size(0), -1)
            decoder3_input1, decoder3_input2, decoder3_input3, decoder3_input4 = encoder_state2_decoder3, encoder_state2_decoder3, encoder_state2_decoder3, encoder_state2_decoder3

        first_name_hidden_output = self.output_layer_action_name_hidden1(decoder3_input1)
        first_name_output = self.output_layer_action_name1(first_name_hidden_output)

        # print('Action last name ...')
        last_name_hidden_output = self.output_layer_action_name_hidden2(decoder3_input2)
        last_name_output = self.output_layer_action_name2(last_name_hidden_output)

        # decoder action flight
        flight_hidden_output = self.output_layer_action_flight_hidden(decoder3_input3)
        flight_output = self.output_layer_action_flight(flight_hidden_output)

        # decoder action state
        state_hidden_output = self.output_layer_action_state_hidden(decoder3_input4)
        state_output = self.output_layer_action_state(state_hidden_output)

        logits_train3 = [first_name_output, last_name_output, flight_output, state_output]

        if args.sql:
            return logits_train1, logits_train2, logits_train3, cond_op_score, cond_col_score, cond_num_score, \
            [cond_str_out_departure_airport_nn, cond_str_out_return_airport_nn, cond_str_out_departure_month_nn, cond_str_out_return_month_nn, \
            cond_str_out_departure_day_nn, cond_str_out_return_day_nn, cond_str_out_departure_time_num_nn, cond_str_out_return_time_num_nn, \
            cond_str_out_class_nn, cond_str_out_price_nn, cond_str_out_num_connections_nn], cond_gate_score, global_pointer
        else:
            return logits_train1, logits_train2, logits_train3
            
    def SQL_forward(self, source_diag, target_diag, size_dialogue, intent, kb, has_reservation, turn_point, col_seq, col_num, gate_label, eval_step, state_tracking_list, kb_true_answer, args):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()

        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)

        ################################################################
        ######################### SQL Generator ########################
        ################################################################
        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_num_col, _ = self.cond_num_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_col, _ = self.cond_col_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_op, _ = self.cond_op_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str, _ = self.cond_str_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str_2, _ = self.cond_str_name_enc_2(col_inp_var) # (b, 12, N_h)
            e_cond_str_3, _ = self.cond_str_name_enc_3(col_inp_var) # (b, 12, N_h)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2
            sql_concat2_embedded_2 = self.sql_encoder_input_projection2_2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded_2 = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded_2, size_dialogue, batch_first=True)
            sql_encoder_outputs2_2, sql_encoder_state2_2 = self._build_encoder2(sql_encoder_emb_inp_padded_2)
            sql_encoder_outputs2_2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2_2, batch_first=True)
            encoder_outputs2_without_kb_2 = sql_encoder_outputs2_2

            # SQLNet
            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze() # (b, s, 1)

            # column num
            h_num_enc = encoder_outputs2_without_kb # (b, s, 256)
            cond_num_score = self.cond_num_out(h_num_enc)

            # Predict the column
            e_cond_col = e_cond_col.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_col_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_col = self.cond_col_att(h_col_enc).expand(-1, -1, c, -1) + e_cond_col # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) + self.cond_col_out_col(e_cond_col)).squeeze() # (b, s, c, 1)
            
            # Predict the operator
            e_cond_op = e_cond_op.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_op_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_op = self.cond_op_att(h_op_enc).expand(-1, -1, c, -1) + e_cond_op # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) + self.cond_op_out_col(e_cond_op)).squeeze() # (b, s, c, 3)

            # Predict the condiction string
            e_cond_str = e_cond_str.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str = self.cond_str_att(h_str_enc).expand(-1, -1, c, -1) + e_cond_str # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc = self.cond_str_out_K(K_cond_str) + self.cond_str_out_col(e_cond_str)
            # Predict the condiction string
            e_cond_str_2 = e_cond_str_2.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_2 = encoder_outputs2_without_kb_2.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_2 = self.cond_str_att_2(h_str_enc_2).expand(-1, -1, c, -1) + e_cond_str_2 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_2 = self.cond_str_out_K_2(K_cond_str_2) + self.cond_str_out_col_2(e_cond_str_2)
            # Predict the condiction string
            e_cond_str_3 = e_cond_str_3.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_3 = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_3 = self.cond_str_att_3(h_str_enc_3).expand(-1, -1, c, -1) + e_cond_str_3 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_3 = self.cond_str_out_K_3(K_cond_str_3) + self.cond_str_out_col_3(e_cond_str_3)

            cond_str_out_departure_airport_nn = self.cond_str_out_departure_airport(cond_str_enc[:, :, 0, :]).squeeze() # (b, s, 50)
            cond_str_out_return_airport_nn = self.cond_str_out_return_airport(cond_str_enc[:, :, 1, :]).squeeze() # (b, s, 50)

            cond_str_out_departure_month_nn = self.cond_str_out_departure_month(cond_str_enc_2[:, :, 2, :]).squeeze() # (b, s, 31)
            cond_str_out_return_month_nn = self.cond_str_out_return_month (cond_str_enc_2[:, :, 3, :]).squeeze() # (b, s, 31)
            cond_str_out_departure_day_nn = self.cond_str_out_departure_day(cond_str_enc_2[:, :, 4, :]).squeeze()
            cond_str_out_return_day_nn = self.cond_str_out_return_day(cond_str_enc_2[:, :, 5, :]).squeeze()

            cond_str_out_departure_time_num_nn = self.cond_str_out_departure_time_num(cond_str_enc_3[:, :, 6, :]).squeeze()
            cond_str_out_return_time_num_nn = self.cond_str_out_return_time_num(cond_str_enc_3[:, :, 7, :]).squeeze()
            cond_str_out_class_nn = self.cond_str_out_class(cond_str_enc_3[:, :, 8, :]).squeeze()
            cond_str_out_price_nn = self.cond_str_out_price(cond_str_enc_3[:, :, 9, :]).squeeze()
            cond_str_out_num_connections_nn = self.cond_str_out_num_connections(cond_str_enc_3[:, :, 10, :]).squeeze() 


        return logits_train1, logits_train2, logits_train3, cond_op_score, cond_col_score, cond_num_score, \
        [cond_str_out_departure_airport_nn, cond_str_out_return_airport_nn, cond_str_out_departure_month_nn, cond_str_out_return_month_nn, \
        cond_str_out_departure_day_nn, cond_str_out_return_day_nn, cond_str_out_departure_time_num_nn, cond_str_out_return_time_num_nn, \
        cond_str_out_class_nn, cond_str_out_price_nn, cond_str_out_num_connections_nn], cond_gate_score

    def C_Inference_bleu(self, intent, size_intent, source_diag, target_diag, size_dialogue, col_seq, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)

        ################################################################
        ######################### Encoder1 #############################
        ################################################################
        # encoder1_intent
        intent_encoder_emb_inp = self.embedding_encoder(intent)
        intent_output, intent_encoder_state1_aux = self.intent_encoder_rnn(intent_encoder_emb_inp)
        intent_encoder_state1_aux = intent_encoder_state1_aux.squeeze(0).unsqueeze(1)
        intent_encoder_state1_aux_expand = intent_encoder_state1_aux.expand(-1, max_seq_len, -1)
        concat1 = torch.cat((encoder_emb_inp, intent_encoder_state1_aux_expand), 2)
        concat1_embedded = self.encoder_input_projection1(concat1)
        concat1_padded_embedded = nn.utils.rnn.pack_padded_sequence(concat1_embedded, size_dialogue, batch_first=True)
        encoder_outputs1, encoder_state1 = self._build_encoder1(concat1_padded_embedded)
        encoder_outputs1, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs1, batch_first=True)
        encoder_outputs1 = encoder_outputs1 + concat1_embedded

        ################################################################
        ######################### Decoder ##############################
        ################################################################

        # decoder 1
        _, logits_train1, sequence_symbols = self.decoder1.Inference_forward(inputs=source_diag,
                              target=target_diag,
                              encoder_hidden=encoder_state1,
                              encoder_outputs=encoder_outputs1,
                              seq_len=size_dialogue)
        _, logits_train1, teacher_sequence_symbols = self.decoder1.Teacher_Inference_forward(inputs=source_diag,
                              target=target_diag[:,:-1],
                              encoder_hidden=encoder_state1,
                              encoder_outputs=encoder_outputs1,
                              seq_len=size_dialogue)

        return logits_train1, sequence_symbols, teacher_sequence_symbols

    def A_Inference_bleu(self, source_diag, target_diag, size_dialogue, has_reservation, col_seq, concat_flight, turn_gate=None, fix_predict_gate=None, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)
        has_reservation_emb = self.embedding_encoder(has_reservation)

        ################################################################
        ######################### SQL Generator ########################
        ################################################################
        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2

            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze(2) # (b, s, 1)
            sigm = nn.Sigmoid()
            cond_gate_prob = sigm(cond_gate_score)
            sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
            predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor)
            if turn_gate is not None:
                predicted_gate[:, turn_gate] = 1. 
            # print('cond_gate_score : ', cond_gate_score.size()) # -9.67
            # print('predicted_gate : ', predicted_gate.size(), predicted_gate[:, -1].item()) # 0.0 (1, s)
            # raise

        ################################################################
        ###################### Encoder2 & Decoder2 #####################
        ################################################################

        # encoder2_init
        has_reservation_emb = has_reservation_emb.unsqueeze(1).expand(-1, max_seq_len, -1)
        concat2 = torch.cat((encoder_emb_inp, has_reservation_emb), 2) # ('kb_encoder_state2_aux_expand : ', (4, 180, 125))
        concat2_embedded = self.encoder_input_projection2(concat2)
        encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(concat2_embedded, size_dialogue, batch_first=True)
        encoder_outputs2, encoder_state2 = self._build_encoder2(encoder_emb_inp_padded)
        encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs2, batch_first=True)
        encoder_outputs2_concat = encoder_outputs2

        ################################################################
        ######################### Concat True KB #######################
        ################################################################
        kb_true_cat = concat_flight.expand(-1, max_seq_len, -1) # (b, s, v)
        if fix_predict_gate is None:
            gate_mask = predicted_gate.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        else:
            gate_mask = fix_predict_gate.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        kb_true_cat_gate = torch.mul(kb_true_cat, gate_mask)
        encoder_outputs2_concat_kb = torch.cat((encoder_outputs2_concat, kb_true_cat_gate), dim=2)
        encoder_outputs2_concat_kb = self.proj_cat_kb(encoder_outputs2_concat_kb)

        ################################################################
        ######################### Decoder ##############################
        ################################################################

        # decoder 2
        _, logits_train2, sequence_symbols = self.decoder2.Inference_forward(inputs=source_diag,
                              target=target_diag,
                              encoder_hidden=encoder_state2,
                              encoder_outputs=encoder_outputs2_concat_kb,
                              seq_len=size_dialogue)
        _, logits_train2, teacher_sequence_symbols = self.decoder2.Teacher_Inference_forward(inputs=source_diag,
                              target=target_diag[:,:-1],
                              encoder_hidden=encoder_state2,
                              encoder_outputs=encoder_outputs2_concat_kb,
                              seq_len=size_dialogue)

        return logits_train2, sequence_symbols, teacher_sequence_symbols, predicted_gate

    def Call_t1_SelfPlayEval(self, intent, size_intent, source_diag, size_dialogue, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)

        ################################################################
        ######################### Encoder1 #############################
        ################################################################
        # encoder1_intent
        intent_encoder_emb_inp = self.embedding_encoder(intent)
        intent_output, intent_encoder_state1_aux = self.intent_encoder_rnn(intent_encoder_emb_inp)
        intent_encoder_state1_aux = intent_encoder_state1_aux.squeeze(0).unsqueeze(1)
        intent_encoder_state1_aux_expand = intent_encoder_state1_aux.expand(-1, max_seq_len, -1)
        concat1 = torch.cat((encoder_emb_inp, intent_encoder_state1_aux_expand), 2)
        concat1_embedded = self.encoder_input_projection1(concat1)
        concat1_padded_embedded = nn.utils.rnn.pack_padded_sequence(concat1_embedded, size_dialogue, batch_first=True)
        encoder_outputs1, encoder_state1 = self._build_encoder1(concat1_padded_embedded)
        encoder_outputs1, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs1, batch_first=True)
        encoder_outputs1 = encoder_outputs1 + concat1_embedded

        ################################################################
        ######################### Decoder ##############################
        ################################################################

        # decoder 1
        _, logits_train1, sequence_symbols = self.decoder1.Inference_forward(inputs=source_diag,
                              target=None,
                              encoder_hidden=encoder_state1,
                              encoder_outputs=encoder_outputs1,
                              seq_len=size_dialogue)

        return logits_train1, sequence_symbols

    def SQL_AND_StateTracking(self, source_diag, size_dialogue, col_seq, args=None):
        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)

        ################################################################
        ######################### SQL Generator ########################
        ################################################################

        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_num_col, _ = self.cond_num_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_col, _ = self.cond_col_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_op, _ = self.cond_op_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str, _ = self.cond_str_name_enc(col_inp_var) # (b, 12, N_h)
            e_cond_str_2, _ = self.cond_str_name_enc_2(col_inp_var) # (b, 12, N_h)
            e_cond_str_3, _ = self.cond_str_name_enc_3(col_inp_var) # (b, 12, N_h)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2
            sql_concat2_embedded_2 = self.sql_encoder_input_projection2_2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded_2 = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded_2, size_dialogue, batch_first=True)
            sql_encoder_outputs2_2, sql_encoder_state2_2 = self._build_encoder2(sql_encoder_emb_inp_padded_2)
            sql_encoder_outputs2_2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2_2, batch_first=True)
            encoder_outputs2_without_kb_2 = sql_encoder_outputs2_2

            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze(2) # (b, s, 1)
            sigm = nn.Sigmoid()
            cond_gate_prob = sigm(cond_gate_score)
            sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
            predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor) # (b, s)

            # column num
            h_num_enc = encoder_outputs2_without_kb # (b, s, 256)
            cond_num_score = self.cond_num_out(h_num_enc)

            # Predict the column
            e_cond_col = e_cond_col.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_col_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_col = self.cond_col_att(h_col_enc).expand(-1, -1, c, -1) + e_cond_col # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) + self.cond_col_out_col(e_cond_col)).squeeze(3) # (b, s, c, 1)
            
            # Predict the operator
            e_cond_op = e_cond_op.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_op_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_op = self.cond_op_att(h_op_enc).expand(-1, -1, c, -1) + e_cond_op # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, 256) + (b, s, c, 256)
            cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) + self.cond_op_out_col(e_cond_op)) # (b, s, c, 3)

            # Predict the condiction string
            e_cond_str = e_cond_str.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str = self.cond_str_att(h_str_enc).expand(-1, -1, c, -1) + e_cond_str # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc = self.cond_str_out_K(K_cond_str) + self.cond_str_out_col(e_cond_str)
            # Predict the condiction string
            e_cond_str_2 = e_cond_str_2.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_2 = encoder_outputs2_without_kb_2.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_2 = self.cond_str_att_2(h_str_enc_2).expand(-1, -1, c, -1) + e_cond_str_2 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_2 = self.cond_str_out_K_2(K_cond_str_2) + self.cond_str_out_col_2(e_cond_str_2)
            # Predict the condiction string
            e_cond_str_3 = e_cond_str_3.unsqueeze(1).expand(-1, max_seq_len, -1, -1) # (b, c, 256) -> (b, s, c, 256)
            h_str_enc_3 = encoder_outputs2_without_kb.unsqueeze(2) # (b, s, 256) -> (b, s, 1, 256)
            K_cond_str_3 = self.cond_str_att_3(h_str_enc_3).expand(-1, -1, c, -1) + e_cond_str_3 # (b, s, 1, 256) -> (b, s, 1, v) -> (b, s, c, v) + (b, s, c, v)
            cond_str_enc_3 = self.cond_str_out_K_3(K_cond_str_3) + self.cond_str_out_col_3(e_cond_str_3)

            cond_str_out_departure_airport_nn = self.cond_str_out_departure_airport(cond_str_enc[:, :, 0, :]) # (b, s, 50)
            cond_str_out_return_airport_nn = self.cond_str_out_return_airport(cond_str_enc[:, :, 1, :]) # (b, s, 50)

            cond_str_out_departure_month_nn = self.cond_str_out_departure_month(cond_str_enc_2[:, :, 2, :])# (b, s, 31)
            cond_str_out_return_month_nn = self.cond_str_out_return_month (cond_str_enc_2[:, :, 3, :])# (b, s, 31)
            cond_str_out_departure_day_nn = self.cond_str_out_departure_day(cond_str_enc_2[:, :, 4, :])
            cond_str_out_return_day_nn = self.cond_str_out_return_day(cond_str_enc_2[:, :, 5, :])

            cond_str_out_departure_time_num_nn = self.cond_str_out_departure_time_num(cond_str_enc_3[:, :, 6, :])
            cond_str_out_return_time_num_nn = self.cond_str_out_return_time_num(cond_str_enc_3[:, :, 7, :])
            cond_str_out_class_nn = self.cond_str_out_class(cond_str_enc_3[:, :, 8, :])
            cond_str_out_price_nn = self.cond_str_out_price(cond_str_enc_3[:, :, 9, :])
            cond_str_out_num_connections_nn = self.cond_str_out_num_connections(cond_str_enc_3[:, :, 10, :])

            return cond_op_score, cond_col_score, cond_num_score, \
            [cond_str_out_departure_airport_nn, cond_str_out_return_airport_nn, cond_str_out_departure_month_nn, cond_str_out_return_month_nn, \
            cond_str_out_departure_day_nn, cond_str_out_return_day_nn, cond_str_out_departure_time_num_nn, cond_str_out_return_time_num_nn, \
            cond_str_out_class_nn, cond_str_out_price_nn, cond_str_out_num_connections_nn], predicted_gate

    def Point_Encode_KB(self, source_diag, size_dialogue, kb, has_reservation, col_seq, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)
        has_reservation_emb = self.embedding_encoder(has_reservation)

        ################################################################
        ###################### Encoder2 & Decoder2 #####################
        ################################################################

        # encoder2_init
        has_reservation_emb = has_reservation_emb.unsqueeze(1).expand(-1, max_seq_len, -1)
        concat2 = torch.cat((encoder_emb_inp, has_reservation_emb), 2) # ('kb_encoder_state2_aux_expand : ', (4, 180, 125))
        concat2_embedded = self.encoder_input_projection2(concat2)
        encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(concat2_embedded, size_dialogue, batch_first=True)
        encoder_outputs2, encoder_state2 = self._build_encoder2(encoder_emb_inp_padded)
        encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs2, batch_first=True)
        encoder_outputs2_concat = encoder_outputs2

        embedded_eval = None
        for b in range(batch_size):
            if embedded_eval is None:
                embedded_eval = encoder_outputs2_concat[b, -1].unsqueeze(0).unsqueeze(1) # (b, s, 256) (256,) (1, 1, 256)
            else:
                embedded_eval = torch.cat((embedded_eval, encoder_outputs2_concat[b, -1].unsqueeze(0).unsqueeze(1)), dim=0)
        global_pointer, _ = self.extKnow.Enough_step_load_memory(kb[:, :, 6:-1], embedded_eval)
        return global_pointer

    def Encode_Flight_KB(self, kb):

        # encoder2_kb
        kb_squeeze = kb.view(-1, 13) # [batch_size*30, 13] <--- [batch_size, 30, 13]
        kb_encoder_emb_inp = self.embedding_encoder(kb_squeeze) # [batch_size*30, 13] ---> [batch_size*30, 13, embedding_encoder_size]
        kb_output0, kb_encoder_state1_aux = self.kb0_encoder_rnn(kb_encoder_emb_inp[:, :-1]) # (b*30, 13, 512) (4, b*30, 256)
        kb_encoder_state1_aux = kb_encoder_state1_aux[-1].squeeze(0)
        kb_encoder_state1_aux = kb_encoder_state1_aux.view(kb.size(0), -1, self.args.kb_hidden_size) # (b, 30, v)
        kb_output0_2, kb_encoder_state1_aux_2 = self.kb0_encoder_rnn(kb_encoder_emb_inp) # (b*30, 13, 512) (4, b*30, 256)
        kb_encoder_state1_aux_2 = kb_encoder_state1_aux_2[-1].squeeze(0)
        kb_encoder_state1_aux_2 = kb_encoder_state1_aux_2.view(kb.size(0), -1, self.args.kb_hidden_size) # (b, 30, v)

        return kb_encoder_state1_aux_2

    def Prior_Gate(self, source_diag, size_dialogue, col_seq):

        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)

        # encoder column
        self.sql_flatten_parameters()
        col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
        col_inp_var = self.embedding_encoder(col_seq)
        col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
        e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

        # sql embedding 
        sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
        sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
        sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
        sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
        encoder_outputs2_without_kb = sql_encoder_outputs2

        # Predict when to send query
        c = self.total_column_K
        e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
        h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
        K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
        cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze(2) # (b, s, 1)
        sigm = nn.Sigmoid()
        cond_gate_prob = sigm(cond_gate_score)
        sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
        predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor) # (b, s)
        return predicted_gate 

    def Call_t2_SelfPlayEvalPrior(self, source_diag, size_dialogue, has_reservation, col_seq, concat_flight, fix_predict_gate=None, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)
        has_reservation_emb = self.embedding_encoder(has_reservation)


        ################################################################
        ######################### SQL Generator ########################
        ################################################################

        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2

            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze(2) # (b, s, 1)
            sigm = nn.Sigmoid()
            cond_gate_prob = sigm(cond_gate_score)
            sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
            predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor) # (b, s)

        ################################################################
        ###################### Encoder2 & Decoder2 #####################
        ################################################################

        # encoder2_init
        has_reservation_emb = has_reservation_emb.unsqueeze(1).expand(-1, max_seq_len, -1)
        concat2 = torch.cat((encoder_emb_inp, has_reservation_emb), 2) # ('kb_encoder_state2_aux_expand : ', (4, 180, 125))
        concat2_embedded = self.encoder_input_projection2(concat2)
        encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(concat2_embedded, size_dialogue, batch_first=True)
        encoder_outputs2, encoder_state2 = self._build_encoder2(encoder_emb_inp_padded)
        encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs2, batch_first=True)
        encoder_outputs2_concat = encoder_outputs2

        ################################################################
        ######################### Concat True KB #######################
        ################################################################

        kb_true_cat = concat_flight.expand(-1, max_seq_len, -1) # (b, s, v)
        if fix_predict_gate is None:
            gate_mask = predicted_gate.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        else:
            gate_mask = fix_predict_gate.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        kb_true_cat_gate = torch.mul(kb_true_cat, gate_mask)
        encoder_outputs2_concat_kb = torch.cat((encoder_outputs2_concat, kb_true_cat_gate), dim=2)
        encoder_outputs2_concat_kb = self.proj_cat_kb(encoder_outputs2_concat_kb)

        ################################################################
        ######################### Decoder ##############################
        ################################################################

        # decoder 2
        _, logits_train2, sequence_symbols = self.decoder2.Inference_forward(inputs=source_diag,
                              target=None,
                              encoder_hidden=encoder_state2,
                              encoder_outputs=encoder_outputs2_concat_kb,
                              seq_len=size_dialogue)

        return logits_train2, sequence_symbols, predicted_gate

    def Call_t2_SelfPlayEval_2(self, source_diag, size_dialogue, has_reservation, col_seq, concat_flight, turn_gate=None, end=0, args=None):

        ################################################################
        ######################### Encoder ##############################
        ################################################################
        batch_size = source_diag.size(0)
        max_seq_len = source_diag.size(1)        
        self.flatten_parameters()
        
        # dynamic_seq2seq / encoder_emb_inp
        encoder_emb_inp = self.embedding_encoder(source_diag) # (b, s, v)
        has_reservation_emb = self.embedding_encoder(has_reservation)

        ################################################################
        ######################### SQL Generator ########################
        ################################################################

        if args.sql:

            # encoder column
            self.sql_flatten_parameters()
            col_seq = col_seq.view(batch_size*self.total_column_K , -1) # (b, 12, name_s) -> (b*12, name_s) since we have to encode column name indiviually
            col_inp_var = self.embedding_encoder(col_seq)
            col_inp_var = col_inp_var.view(batch_size, self.total_column_K, -1)
            e_cond_gate, _ = self.cond_gate_name_enc(col_inp_var) # (b, 12, N_h)

            # sql embedding 
            sql_concat2_embedded = self.sql_encoder_input_projection2(encoder_emb_inp) 
            sql_encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(sql_concat2_embedded, size_dialogue, batch_first=True)
            sql_encoder_outputs2, sql_encoder_state2 = self._build_encoder2(sql_encoder_emb_inp_padded)
            sql_encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(sql_encoder_outputs2, batch_first=True)
            encoder_outputs2_without_kb = sql_encoder_outputs2

            # Predict when to send query
            c = self.total_column_K
            e_cond_gate = e_cond_gate.sum(1).unsqueeze(1).expand(-1, max_seq_len, -1) # (b, c, 256) -> (b, 256) -> (b, s, 256)
            h_gate_enc = encoder_outputs2_without_kb # (b, s, 256)
            K_cond_gate = self.cond_gate_att(h_gate_enc) + e_cond_gate # (b, s, 256) + (b, s, 256)
            cond_gate_score = self.cond_gate_out(self.cond_gate_out_K(K_cond_gate) + self.cond_gate_out_col(e_cond_gate)).squeeze(2) # (b, s, 1)
            sigm = nn.Sigmoid()
            cond_gate_prob = sigm(cond_gate_score)
            sigmoid_matrix = torch.ones_like(cond_gate_prob) * 0.5
            predicted_gate = torch.gt(cond_gate_prob, sigmoid_matrix).type(torch.cuda.FloatTensor) # (b, s)
            if turn_gate is not None:
                predicted_gate[:, turn_gate] = 1.   

        ################################################################
        ###################### Encoder2 & Decoder2 #####################
        ################################################################

        # encoder2_init
        has_reservation_emb = has_reservation_emb.unsqueeze(1).expand(-1, max_seq_len, -1)
        concat2 = torch.cat((encoder_emb_inp, has_reservation_emb), 2) # ('kb_encoder_state2_aux_expand : ', (4, 180, 125))
        concat2_embedded = self.encoder_input_projection2(concat2)
        encoder_emb_inp_padded = nn.utils.rnn.pack_padded_sequence(concat2_embedded, size_dialogue, batch_first=True)
        encoder_outputs2, encoder_state2 = self._build_encoder2(encoder_emb_inp_padded)
        encoder_outputs2, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs2, batch_first=True)
        encoder_outputs2_concat = encoder_outputs2

        # decoder 2
        kb_true_cat = concat_flight.expand(-1, max_seq_len, -1) # (b, s, v)
        gate_mask = predicted_gate.unsqueeze(2).expand(-1, -1, kb_true_cat.size(2)).type(torch.cuda.FloatTensor)
        kb_true_cat_gate = torch.mul(kb_true_cat, gate_mask)
        encoder_outputs2_concat_kb = torch.cat((encoder_outputs2_concat, kb_true_cat_gate), dim=2)
        encoder_outputs2_concat_kb = self.proj_cat_kb(encoder_outputs2_concat_kb)

        ################################################################
        ######################### Decoder ##############################
        ################################################################

        # decoder 2
        _, logits_train2, sequence_symbols = self.decoder2.Inference_forward(inputs=source_diag,
                              target=None,
                              encoder_hidden=encoder_state2,
                              encoder_outputs=encoder_outputs2_concat_kb,
                              seq_len=size_dialogue)

        if end == 0:
            return logits_train2, sequence_symbols, predicted_gate
        else:
            ################################################################
            ######################### Final Status #########################
            ################################################################
            if args.action_att:

                timestep = encoder_outputs2_concat.size(1)

                encoder_outputs2_concat_action = encoder_outputs2_concat
                embedded = encoder_state2[-1].unsqueeze(0).transpose(0, 1) # (2, b, 256) -> (1, b, 256) -> (b, 1, 256) 
                v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [b, 1, 256]
                hidden = embedded
                hidden = hidden.expand(-1, timestep, -1)

                energy1 = F.relu(self.decoder3_attention_linear1(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
                # energy2 = F.relu(self.decoder3_attention_linear2(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
                # energy3 = F.relu(self.decoder3_attention_linear3(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
                # energy4 = F.relu(self.decoder3_attention_linear4(torch.cat([hidden, encoder_outputs2_concat_action], dim=2))).transpose(1, 2)  # [b, 256, s]
                
                action_mask = torch.arange(timestep)[None, :].cuda() < size_dialogue[:, None] ; action_mask = action_mask.cuda()
                action_mask = ~action_mask
                attn_weights1 = F.softmax(torch.bmm(v, energy1).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
                # attn_weights2 = F.softmax(torch.bmm(v, energy2).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
                # attn_weights3 = F.softmax(torch.bmm(v, energy3).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
                # attn_weights4 = F.softmax(torch.bmm(v, energy4).squeeze(1).masked_fill(action_mask, -np.inf), dim=1).unsqueeze(1) # [b, 1, s] -> [b, s] # [b, 1, s]
                context1 = attn_weights1.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
                # context2 = attn_weights2.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
                # context3 = attn_weights3.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
                # context4 = attn_weights4.bmm(encoder_outputs2_concat_action)  # [b, 1, s] * [b, s, 256] ->[b, 1, 256]
                decoder3_input1 = torch.cat([embedded, context1], 2).squeeze(1)
                decoder3_input2 = torch.cat([embedded, context1], 2).squeeze(1)
                decoder3_input3 = torch.cat([embedded, context1], 2).squeeze(1) # [b, 512]
                decoder3_input4 = torch.cat([embedded, context1], 2).squeeze(1)
                
            else:
                encoder_state2_decoder3 = encoder_state2[-1].squeeze(0)
                encoder_state2_decoder3 = encoder_state2_decoder3.view(encoder_state2_decoder3.size(0), -1)
                decoder3_input1, decoder3_input2, decoder3_input3, decoder3_input4 = encoder_state2_decoder3, encoder_state2_decoder3, encoder_state2_decoder3, encoder_state2_decoder3

            first_name_hidden_output = self.output_layer_action_name_hidden1(decoder3_input1)
            first_name_output = self.output_layer_action_name1(first_name_hidden_output)

            # print('Action last name ...')
            last_name_hidden_output = self.output_layer_action_name_hidden2(decoder3_input2)
            last_name_output = self.output_layer_action_name2(last_name_hidden_output)

            # decoder action flight
            flight_hidden_output = self.output_layer_action_flight_hidden(decoder3_input3)
            flight_output = self.output_layer_action_flight(flight_hidden_output)

            # decoder action state
            state_hidden_output = self.output_layer_action_state_hidden(decoder3_input4)
            state_output = self.output_layer_action_state(state_hidden_output)

            logits_train3 = [first_name_output, last_name_output, flight_output, state_output]
            return logits_train3
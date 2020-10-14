import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, num_layer, vocab_size, max_len, hidden_size, sos_id, eos_id, batch_size, att_method, att_mlp, att_type, n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = None
        self.num_layer = num_layer

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = None
        self.attention = Attention(batch_size, hidden_size, att_method, att_mlp)
        self.att_type = att_type
        self.out = None # nn.Linear(self.hidden_size, self.output_size)
        self.use_attention = use_attention

    def forward_step(self, input_var, hidden, encoder_outputs, seq2seq_attention_mask, function, action=None):
        input_var = input_var.unsqueeze(1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        # print('\nDecoderRNN.py -- [forward_step] ')
        # print('input_var.size() : ', input_var.size())
        # print('self.num_layer : ', self.num_layer)
        # print('hidden.size() : ', hidden.size())
        # print('encoder_outputs.size() : ', encoder_outputs.size(), '\n')
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        attn = None
        if self.use_attention and self.att_type == 'Bahd':
            # print('Use Bahd forward !')
            # print('input embedded.size() : ', embedded.size()) # (b, 1, 256)
            # print('hidden : ', hidden.size()) # (2, b, 256)
            # print('encoder_outputs : ', encoder_outputs.size()) # (b, s, 256)
            output, hidden, weights = self.attention.Bahd_forward(embedded, hidden, encoder_outputs, seq2seq_attention_mask, self.rnn, self.out)
            attn = weights
        else:
            # print('input embedded.size() : ', embedded.size()) # (b, 1, 256)
            # print('hidden : ', hidden.size()) # (2, b, 256)
            # print('encoder_outputs : ', encoder_outputs.size()) # (b, s, 256)
            output, hidden = self.rnn(embedded, hidden)
            output = self.out(output.contiguous().view(-1, self.hidden_size)).view(batch_size, output_size, -1)
        # print('output : ', output.size())
        # print('predicted : ', predicted.size())
        # raise
        return output, hidden, attn

    def Inference_forward_step(self, input_var, hidden):
        input_var = input_var.unsqueeze(1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        # print('embedded : ', embedded.size())
        # print('hidden : ', hidden.size())
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.contiguous().view(-1, self.hidden_size)).view(batch_size, output_size, -1)
        return output, hidden

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, turn_point=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, accerlator=True):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # print('\nDecoderRNN.py -- [forward] ')
        # print('encoder_hidden.size() : ', encoder_hidden.size())
        # print('encoder_outputs.size() : ', encoder_outputs.size(), '\n')
        action_ = None
        inputs, _, _ = self._validate_args(inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio)
        if accerlator == False: # action
            action_ = True
            decoder_hidden = encoder_hidden[-1]
            decoder_hidden = decoder_hidden.unsqueeze(0)
        else:
            action_ = False
            decoder_hidden = self._init_state(encoder_hidden)
        # if self.num_layer > 1:
        #    decoder_hidden = decoder_hidden.repeat(2, 1, 1)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        use_teacher_forcing = True
        b, s = encoder_outputs.size(0), encoder_outputs.size(1)
        seq2seq_attention_mask = torch.zeros((b, s)).type(torch.FloatTensor).cuda()
        seq2seq_attention_mask[:, 0] = 1
        each_step_mask = torch.zeros((b, s)).type(torch.FloatTensor).cuda()
        if use_teacher_forcing:
            tensor_decoder_output = None
            decoder_input = inputs
            # print('inputs : ', inputs.size())
            # print('encoder_outputs : ', encoder_outputs.size())
            for di in range(decoder_input.size(1)):
                # print('inputs.size(1) : ', inputs.size(1), ' di : ', di)
                # print('encoder_outputs : ', encoder_outputs.size())
                # print('turn_point : ', turn_point.size())
                # print('decoder_hidden : ', decoder_hidden.size())
                each_step_mask[:, :(di+1)] = 1 # since di=0 ~ s-1
                # print('each_step_mask : ', each_step_mask.size(), each_step_mask)
                # print('turn point : ', turn_point)
                # print('turn point mask : ', turn_point[:, di].unsqueeze(1).repeat(1, s).type(torch.FloatTensor).cuda().size(), turn_point[:, di].unsqueeze(1).repeat(1, s).type(torch.FloatTensor).cuda())
                seq2seq_attention_mask = torch.mul(each_step_mask, turn_point[:, di].unsqueeze(1).repeat(1, s).type(torch.FloatTensor).cuda()) + seq2seq_attention_mask.type(torch.cuda.FloatTensor)
                seq2seq_attention_mask = torch.gt((seq2seq_attention_mask), 0).cuda()
                # print('di : ', di)
                # print('seq2seq_attention_mask : ', seq2seq_attention_mask[:, :(di+1)].size(), seq2seq_attention_mask[:, :(di+1)])
                # print('turn point : ', turn_point[:, :(di+1)].size(), turn_point[:, :(di+1)])

                turn = turn_point[:, di].unsqueeze(1).unsqueeze(0).repeat(decoder_hidden.size(0), 1, decoder_hidden.size(2)).type(torch.FloatTensor).cuda()
                encoder_outputs_resize = encoder_outputs[:, di, :].squeeze(1).unsqueeze(0).repeat(decoder_hidden.size(0), 1, 1)
                # print('turn : ', turn.size())
                # print('encoder_outputs_resize : ', encoder_outputs_resize.size())
                # print('turn : ', turn.type())
                # print('encoder_outputs_resize : ', encoder_outputs_resize.type())
                if accerlator : 
                    decoder_hidden_new = (1. - turn) * decoder_hidden + turn * encoder_outputs_resize
                else:
                    decoder_hidden_new = decoder_hidden
                # print('decoder_hidden_new : ', decoder_hidden_new.size())
                
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input[:,di], decoder_hidden_new, encoder_outputs[:, :(di+1)], seq2seq_attention_mask[:, :(di+1)], function=function, action=action_)
                if tensor_decoder_output is None:
                    tensor_decoder_output = decoder_output
                else:
                    tensor_decoder_output = torch.cat((tensor_decoder_output, decoder_output), 1)
                # step_output = decoder_output.unsqueeze(1)
                # decoder_outputs.append(step_output)
            # print('End Decoder ... ')
            # print('*'*100)

        # else:
            
        #     decoder_input = inputs[:, 0].unsqueeze(1)
        #     for di in range(max_length):
        #         decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
        #                                                                  function=function)
        #         print('<'*100)
        #         print('decoder_output : ', decoder_output.size())
        #         print('decoder_hidden : ', decoder_hidden.size())
        #         print('<'*100)
        #         step_output = decoder_output.squeeze(1)
        #         symbols = decode(di, step_output, step_attn)
        #         decoder_input = symbols
        # print('+'*100)
        # print('decoder_outputs : ', len(decoder_outputs))
        # print('decoder_hidden : ', decoder_hidden.size())
        # print('+'*100)
        return decoder_outputs, tensor_decoder_output

    def Inference_forward(self, inputs=None, target=None, encoder_hidden=None, encoder_outputs=None, seq_len=None):

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, 0.0)
        decoder_hidden = encoder_outputs[:, -1, :].squeeze(1).unsqueeze(0).repeat(encoder_hidden.size(0), 1, 1)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols.cpu().view(-1).numpy())

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        b, s = encoder_outputs.size(0), encoder_outputs.size(1)

        decoder_input = inputs[:, 0]
        tensor_decoder_output = None
        for di in range(self.max_length):
            decoder_hidden_new = decoder_hidden
            # print('decoder_input : ', decoder_input.size())
            # print('decoder_hidden_new : ', decoder_hidden_new.size())
            decoder_output, decoder_hidden = self.Inference_forward_step(decoder_input, decoder_hidden_new)
            if tensor_decoder_output is None:
                tensor_decoder_output = decoder_output
            else:
                tensor_decoder_output = torch.cat((tensor_decoder_output, decoder_output), 1)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output)
            decoder_input = symbols.squeeze(1) # (b, 1) -> (b,)
        return decoder_outputs, tensor_decoder_output, sequence_symbols

    def Teacher_Inference_forward(self, inputs=None, target=None, encoder_hidden=None, encoder_outputs=None, seq_len=None):

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, 0.0)
        decoder_hidden = encoder_outputs[:, -1, :].squeeze(1).unsqueeze(0).repeat(encoder_hidden.size(0), 1, 1)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols.cpu().view(-1).numpy())

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        b, s = encoder_outputs.size(0), encoder_outputs.size(1)
        # teacher
        decoder_input = target
        tensor_decoder_output = None
        for di in range(decoder_input.size(1)):
            decoder_hidden_new = decoder_hidden
            # print('decoder_input : ', decoder_input.size())
            # print('decoder_hidden_new : ', decoder_hidden_new.size())
            decoder_output, decoder_hidden = self.Inference_forward_step(decoder_input[:,di], decoder_hidden_new)
            if tensor_decoder_output is None:
                tensor_decoder_output = decoder_output
            else:
                tensor_decoder_output = torch.cat((tensor_decoder_output, decoder_output), 1)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output)
        return decoder_outputs, tensor_decoder_output, sequence_symbols

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):

        # inference batch size
        batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            inputs = inputs.cuda()
        max_length = self.max_length

        return inputs, batch_size, max_length

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Attention(nn.Module):
    def __init__(self, batch_size, hidden_size, method="concat", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.va.size(0))
            self.va.data.uniform_(-stdv, stdv)
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.va.size(0))
            self.va.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def attention_forward(self, last_hidden, encoder_outputs, seq2seq_attention_mask):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden) # [1, b, v] -> phi:v*v -> [1, b, v]
            encoder_outputs = self.psi(encoder_outputs) # [b, s v] -> psi:v*v -> [b, s, v]

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)
        seq2seq_attention_mask_fill = ~seq2seq_attention_mask
        # print('seq2seq_attention_mask_fill : ', seq2seq_attention_mask_fill.size(), seq2seq_attention_mask_fill)
        seq2seq_attention_mask_fill = seq2seq_attention_mask_fill.unsqueeze(1) # [b, s] -> [b, 1, s]
        attention_energies.masked_fill(seq2seq_attention_mask_fill, -np.inf) # [b, 1, s]

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):

        if method == 'dot':
            last_hidden = last_hidden.transpose(0, 1) # [1, b, v] -> [b, 1, v]
            encoder_outputs = encoder_outputs.transpose(1, 2) # [b, s, v] -> [b, v, s]
            return last_hidden.bmm(encoder_outputs) # [b, 1, v] * [b, v, s] -> [b, 1, s]

        elif method == 'general':
            last_hidden = self.Wa(last_hidden.transpose(0, 1)) # [1, b, v] -> [b, 1, v]
            encoder_outputs = encoder_outputs.transpose(1, 2) # [b, s, v] -> [b, v, s]
            return last_hidden.bmm(encoder_outputs) # [b, 1, v] * [b, v, s] -> [b, 1, s]

        elif method == "concat":

            timestep = encoder_outputs.size(1)
            v = self.va.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [b, 1, 256]
            last_hidden = last_hidden.transpose(0, 1) # [1, b, v] -> [b, 1, v]
            last_hidden = last_hidden.expand(-1, timestep, -1) # [b, 1, v] -> [b, s, v]
            energy = F.relu(self.Wa(torch.cat([last_hidden, encoder_outputs], dim=2))).transpose(1, 2)  # [b, 256, s]
            attn_weights = F.softmax(torch.bmm(v, energy).squeeze(1), dim=1).unsqueeze(1) # [b, 1, 256]*[b, 256, s] - > [b, 1, s] -> [b, s] # [b, 1, s]
            return attn_weights

        elif method == "bahdanau":
            timestep = encoder_outputs.size(1)
            v = self.va.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [b, 1, 256]
            last_hidden = last_hidden.transpose(0, 1) # [1, b, v] -> [b, 1, v]
            last_hidden = last_hidden.expand(-1, timestep, -1) # [b, 1, v] -> [b, s, v]
            energy = F.tahn(self.Wa(last_hidden) + self.Ua(encoder_outputs)).transpose(1, 2)  # [b, s, v] + [b, s, v] -> [b, v, s]
            attn_weights = F.softmax(torch.bmm(v, energy).squeeze(1), dim=1).unsqueeze(1) # [b, 1, v]*[b, v, s] - > [b, 1, s] -> [b, s] # [b, 1, s]
            return attn_weights

        else:
            raise NotImplementedError
   
    def Bahd_forward(self, embedded, prev_hidden, encoder_outputs, seq2seq_attention_mask, rnn, proj_out):

        # Attention weights
        last_hidden = prev_hidden[-1].unsqueeze(0)
        weights = self.attention_forward(last_hidden, encoder_outputs, seq2seq_attention_mask)  # [b, 1, s]
        context = weights.bmm(encoder_outputs) # [b, 1, s] * [b, s, v] -> [b, 1, v]

        rnn_input = torch.cat([embedded, context], 2) # [b, 1, v] [b, 1, v] > [b, 1, 2*v]

        outputs, hidden = rnn(rnn_input, prev_hidden) # [b, 1, v] [1, b, v]

        # print('outputs : ', outputs.size()) # (b, 1, v)
        # print('context : ', context.size()) # (b, 1, v)
        outputs = proj_out(torch.cat((outputs, context), 2))

        return outputs, hidden, weights

    # def Luong_forward(self, embedded, prev_hidden, prev_attention, encoder_outputs, seq2seq_attention_mask, rnn, proj_out):

    #     # RNN (Eq 7 paper)
    #     embedded = self.embedding(input).unsqueeze(1) # [B, H]
    #     prev_hidden = prev_hidden.unsqueeze(0)
    #     rnn_input = torch.cat((embedded, prev_attention), -1) # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
    #     rnn_output, hidden = rnn(rnn_input.transpose(1, 0), prev_hidden)
    #     rnn_output = rnn_output.squeeze(1)

    #     # Attention weights (Eq 6 paper)
    #     weights = self.attention.forward(rnn_output, encoder_outputs, seq_len) # B x T
    #     context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x N]

    #     # Projection (Eq 8 paper)
    #     # /!\ Don't apply tanh on outputs, it fucks everything up
    #     outputs = proj_out(torch.cat((outputs, context), 1))

    #     return output, hidden, weights

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ExternalKnowledge(nn.Module):
    # https://github.com/jojonki/MemoryNetworks/blob/master/memnn.py
    def __init__(self, vocab, embedding_dim, hop, dropout, args):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
            # equal to  self.C_{} = nn.Embedding(vocab, embedding_dim)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def Enough_step_load_memory(self, story, hidden):
        
        # Forward multiple hop mechanism
        u = [hidden.squeeze(1)] # (b, 1, 256) > (b, 256)
        story_size = story.size() # (b, nb, 12) <> (b, m, s)
        self.m_story = []

        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1)) # (b, m, s) > (b, m*s) > (b, m*s, e)
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # (b, m, s, e)
            embed_A = torch.sum(embed_A, 2) # (b, m, e) Bag-of-word representation
            # embed_A = self.dropout_layer(embed_A)
            
            # if(len(list(u[-1].size()))==1): 
            #     u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A) # (b, 256) > (b, 1, 256) > (b, m, 256)
            prob_logit = torch.sum(embed_A*u_temp, 2) # (b, m, e) * (b, m, 256) > (b, m)
            prob_   = self.softmax(prob_logit)
            
            embed_C = self.C[hop+1](story.contiguous().view(story_size[0], -1)) # (b, m, s) > (b, m*s) > (b, m*s, e)
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) # (b, m, s, e)
            embed_C = torch.sum(embed_C, 2) # (b, m, e) Bag-of-word representation

            prob = prob_.unsqueeze(2).expand_as(embed_C) # (b, m) > (b, m, 1) > (b, m, e)
            o_k  = torch.sum(embed_C*prob, 1) # (b, m, e) * (b, m, 256) > (b, m)
            u_k = u[-1] + o_k # (b, 256) + (b, m)
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return prob_logit, u[-1]

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
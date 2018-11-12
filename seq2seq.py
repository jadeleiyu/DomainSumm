from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import cPickle as pkl
use_cuda = torch.cuda.is_available()
import helper
import math

MAX_LENGTH = 150
class Attention(nn.Module):
     # modified from
     # https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.key = nn.Linear(dim, dim, bias=False)
        self.mask = None

    def set_mask(self, mask):
        if mask is None:
            self.mask = None
        else:
            self.mask = (mask[:, None, :] != 1.).data

    def forward(self, input, context):
        batch_size, input_size, hidden_size = input.size()
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        key = self.key(input)
        attn = torch.bmm(key, context.transpose(1, 2)) / math.sqrt(self.dim)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -1e7)
        #attn = F.softmax(attn, dim=-1)
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)
        # (batch, out_len, in_clen) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix_c = torch.bmm(attn, context)
        self.att_dist = attn
        return mix_c, attn

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, embedding=None):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_()), \
               Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_())


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, embedding=None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.attn = Attention(hidden_size)
        self.attn_MLP = nn.Sequential(nn.Linear(hidden_size*2, embedding_dim),
                                      nn.Tanh())
        self.out = nn.Linear(embedding_dim, 50000)
        self.out.weight.data = self.embedding.weight.data[:50000]


    def forward(self, input, hidden, encoder_outputs):
        bsz, nsteps = input.size() # nsteps - summary step
        _, encsteps, _ = encoder_outputs.size()
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)

        # key = self.attn_Projection(output)  # bsz x nsteps x nhid
        # logits = torch.bmm(key, encoder_outputs.transpose(1,2)) # bsz x nsteps x encsteps
        # attn_weights = F.softmax(logits.view(-1, encsteps)).view(bsz, nsteps, encsteps) # bsz x nsteps x encsteps
        # attn_applied = torch.bmm(attn_weights, encoder_outputs) # bsz x nsteps x nhid

        mix,attn_weights = self.attn(output,encoder_outputs)
        output = torch.cat((output, mix), 2).view(-1, self.hidden_size*2)  # bsz*nsteps x nhid*2
        output = self.attn_MLP(output)

        output = F.log_softmax(self.out(output))

        ### add pointer
        # p_pointer =
        return output, hidden, attn_weights

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()), \
               Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())


class AttnSeq2Seq(nn.Module):
    def __init__(self, config, n_layers=2):
        super(AttnSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if not(config.pretrained_embedding is None):
            self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

        self.encoder = EncoderRNN(config.vocab_size, config.embedding_dim, config.word_GRU_hidden_units,
                                  n_layers, self.embedding)
        self.decoder = AttnDecoderRNN(config.vocab_size, config.embedding_dim, config.word_GRU_hidden_units * 2,
                                      n_layers, self.embedding)

    def forward(self, input, output):
        hidden = self.encoder.initHidden(input.size(0))
        encoder_outputs, hidden = self.encoder(input, hidden)
        h, c = hidden
        h = torch.cat([h[0], h[1]], dim=1)[None, :, :]
        c = torch.cat([c[0], c[1]], dim=1)[None, :, :]
        hidden = (h, c)
        logp, _, attn_weights = self.decoder(output, hidden, encoder_outputs)
        return logp

    def beamsearch(self, input, id2word, max_length=MAX_LENGTH, beam_size=5):
        hidden = self.encoder.initHidden(1)
        encoder_outputs, hidden = self.encoder(input[None, :], hidden)

        h, c = hidden
        h = torch.cat([h[0], h[1]], dim=1)[None, :, :]
        c = torch.cat([c[0], c[1]], dim=1)[None, :, :]
        hidden = (h, c)

        # init for beam search
        best_k_seq = [[BOS]]
        best_k_probs = [0.]
        best_k_hiddens = [hidden]

        for di in range(max_length):
            best_k_seq, best_k_probs, best_k_hiddens = beam_expand(best_k_seq, best_k_probs, best_k_hiddens,
                                                                   encoder_outputs, self.decoder, id2word, beam_size)
        # find final best output
        index = np.argsort(best_k_probs)[::-1][0]
        best_seq = best_k_seq[index]
        best_seq = best_seq[1:]
        decoded_words = ' '.join([id2word[i] for i in best_seq])
        return decoded_words


def beam_expand(best_k_seq, best_k_probs, best_k_decoder_hiddens, encoder_outputs, decoder, id2word, beam_size):
    """
    :param best_k_seq: list of beam_size of best sequences
    :param best_k_probs: probs corresponding to the best k sequences
    :return: next_best_k*k_seq,next_best_k*k_probs
    """
    next_best_k_squared_seq = []
    next_best_k_squared_probs = []
    next_best_k_squared_decoder_hiddens = []

    for b in range(len(best_k_seq)):
        seq = best_k_seq[b]
        prob = best_k_probs[b]
        decoder_hidden = best_k_decoder_hiddens[b]
        if id2word[seq[-1]] == u'<eos>':
            #if end of token, make sure no children
            next_best_k_squared_seq.append(seq)
            next_best_k_squared_probs.append(prob)
            next_best_k_squared_decoder_hiddens.append(decoder_hidden)
        else:
            # append the top k children
            decoder_input = Variable(torch.LongTensor([[seq[-1]]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, _= decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output = decoder_output.squeeze()
            decoder_output[UNK] = -1000000.
            decoder_output[EOS] = -1000000.
            topv, topi = decoder_output.data.topk(beam_size)
            for i in range(beam_size):
                next_seq = seq[:]
                next_seq.append(topi[i])

                next_best_k_squared_seq.append(next_seq)
                next_best_k_squared_probs.append(prob + topv[i])
                next_best_k_squared_decoder_hiddens.append(decoder_hidden)
    # contract to the best k
    indexs = np.argsort(next_best_k_squared_probs)[::-1][:beam_size]
    beam_seqs = [next_best_k_squared_seq[i] for i in indexs]
    beam_probs = [next_best_k_squared_probs[i] for i in indexs]
    beam_hiddens = [next_best_k_squared_decoder_hiddens[i] for i in indexs]
    return beam_seqs, beam_probs, beam_hiddens
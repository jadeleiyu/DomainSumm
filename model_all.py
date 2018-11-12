# coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

torch.manual_seed(233)

class FullyShare(nn.Module):
    def __init__(self, config):
        super(FullyShare, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.category_size = config.category_size
        self.category_dim = config.category_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        #self.category_embedding = nn.Embedding(self.category_size, self.category_dim)
        self.category_embedding = nn.Linear(self.category_size, self.category_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # self.encoder = nn.Sequential(nn.Linear(800, 400),
        #                              nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(400+self.category_dim, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x, c):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # category embedding
        #print("c:", c)
        cat_feature = self.category_embedding(c)

        # word level LSTM
        #print("x:", x)
        #print("shape of x:", x.size())
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        #print("word_features:", word_features)
        #print("shape of word_features:", word_features.size())
        word_outputs, h = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        #print("word_outputs:", word_outputs)
        #print("shape of word_outputs:", word_outputs.size())
        #print("h:", h)
        #print("shape of h[0]:", h[0].size())
        #print("shape of h[1]:", h[1].size())
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)
        #print("sent_features:", sent_features)
        #print("shape of sent_features:", sent_features.size())

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)
        #print("enc_output:", enc_output)
        #print("shape of enc_output:", enc_output.size())
        #print("cat_feature:", cat_feature)
        # inp_list=torch.unbind(enc_output,dim=1)
        # enc_cat_output= torch.cat([torch.cat(inp, cat_feature) for inp in inp_list],dim=1)
        expanded_cat = cat_feature.expand(enc_output.size(1), cat_feature.size(0))
        enc_cat_output =torch.cat([enc_output, expanded_cat.reshape(1, expanded_cat.size(0), -1)], 2)
        # concat cat_feature and enc_output
        prob = self.decoder(enc_cat_output).squeeze(0)

        return prob.view(sequence_num, 1)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)
        return enc_output


class PrivateShare(nn.Module):
    def __init__(self, config):
        super(PrivateShare, self).__init__()

        # Parameters
        # self.vocab_size = config.vocab_size
        # self.embedding_dim = config.embedding_dim
        self.category_size = config.category_size
        # self.category_dim = config.category_dim
        # self.word_input_size = config.word_input_size
        # self.sent_input_size = config.sent_input_size
        # self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        # self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        self.encoder_list = nn.ModuleList()
        for i in range(self.category_size+1): # one extra for sharing
            self.encoder_list.append(Encoder(config))


        self.decoder_list = nn.ModuleList()
        for i in range(self.category_size):
            self.decoder_list.append(nn.Sequential(nn.Linear(400*2, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid()))

    def forward(self, x, c):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        c_number = c.data.cpu().numpy()[0]
        #print("c_number:", c_number)
        private_output = self.encoder_list[c_number].forward(x)
        share_output = self.encoder_list[-1].forward(x)

        #concat private_output and shareoutput
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes
        enc_output = torch.cat([private_output,share_output],dim=2)
        prob = self.decoder_list[c_number](enc_output)
        return prob.view(sequence_num, 1)


class DomainModel(nn.Module):
    def __init__(self, config):
        super(DomainModel, self).__init__()

        # Parameters
        # self.vocab_size = config.vocab_size
        # self.embedding_dim = config.embedding_dim
        self.category_size = config.category_size
        # self.category_dim = config.category_dim
        # self.word_input_size = config.word_input_size
        # self.sent_input_size = config.sent_input_size
        # self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        # self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        self.encoder_list = nn.ModuleList()
        for i in range(self.category_size): # one extra for sharing
            self.encoder_list.append(Encoder(config))


        self.decoder_list = nn.ModuleList()
        for i in range(self.category_size):
            self.decoder_list.append(nn.Sequential(nn.Linear(400, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid()))

    def forward(self, x, c):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        c_number = c.data.cpu().numpy()[0]
        print("c_number:", c_number)
        enc_output = self.encoder_list[c_number].forward(x)

        #concat private_output and shareoutput
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes
        prob = self.decoder_list[c_number](enc_output)
        return prob.view(sequence_num, 1)


class GeneralModel(nn.Module):
    def __init__(self, config):
        super(GeneralModel, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.category_size = config.category_size
        self.category_dim = config.category_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.category_embedding = nn.Embedding(self.category_size, self.category_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Sequential(nn.Linear(800, 400),
                                     nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(400, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x, c):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)

        prob = self.decoder(enc_output)

        return prob.view(sequence_num, 1)
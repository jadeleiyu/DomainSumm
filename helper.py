# coding:utf8
import pickle as pkl
import logging
import random
from collections import namedtuple
from copy import deepcopy

import numpy
import torch
from torch.autograd import Variable

random.seed(1234)

# os.chdir('/home/ml/ydong26/Dropbox/summarization_RL/summarization_RL/src/')
Config = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'category_size', 'category_dim',
                     'word_input_size', 'sent_input_size',
                     'word_GRU_hidden_units', 'sent_GRU_hidden_units',
                     'pretrained_embedding', 'word2id', 'id2word'])


class Document():
    def __init__(self, content, label, summary):
        self.content = content
        self.label = label
        self.summary = summary


# class Dataset():
#     def __init__(self, data_list):
#         self._data = data_list
#
#     def __len__(self):
#         return len(self._data)
#
#     def __call__(self, batch_size, shuffle=True):
#         max_len = len(self)
#         if shuffle:
#             random.shuffle(self._data)
#         batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
#         return batchs
#
#     def __getitem__(self, index):
#         return self._data[index]


# a bunch of converter functions
def tokens_to_sentences(token_list):
    # convert a token list to sents list
    # this is a cheap fix, might need better way to do it
    sents_list = []
    counter = 0
    for i, token in enumerate(token_list):
        if token == '.' or token == '!' or token == '?':
            sents_list.append(token_list[counter:i + 1])  # include .!? in sents
            counter = i + 1
    sents_list = [" ".join(s) for s in sents_list]

    sents_list = [s.replace("<s>", '') for s in sents_list]
    sents_list = [s.replace("</s>", '') for s in sents_list]

    # sequence = " ".join(token_list).strip()
    # sequence = sequence.replace("\\","")
    # if "<s>" not in token_list:
    #     extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'u.s']
    #     sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #     sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    #     new_list = sent_tokenize(sequence)
    #     new_list = [s for s in new_list if len(s.split())>1] #all <s> and </s> is removed
    #     #print(new_list)
    # else:
    #     new_list = sequence.split("</s>")
    #     new_list = [s+"</s>" for s in new_list if len(s.split()) > 1]
    #
    #     new_list = [s.replace("<s>",'') for s in new_list]
    #     new_list = [s.replace("</s>", '') for s in new_list]
    return sents_list


def remove_control_tokens(text):
    if type(text) == str:
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
    # list of strings
    if type(text) == list:
        text = [s.replace("<s>", "") for s in text if type(s) == str]
        text = [s.replace("</s>", "") for s in text if type(s) == str]
    return text


def prepare_data(doc, word2id):
    V = len(word2id)
    print("in prepare_data, V =", V)
    data = deepcopy(doc.content)
    max_len = -1  # this is for padding
    for words in data:
        max_len = max(max_len, len(words))

    sent_list = []
    one_hot_sent_list = []
    for words in data:
        sent = [word2id[word] if word in word2id else 1 for word in words]
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        one_hot_sent = [one_hot(idx, V) for idx in sent]
        sent_list.append(sent)
        one_hot_sent_list.append(one_hot_sent)

    sent_array = numpy.array(sent_list)
    one_hot_sent_array = numpy.array(one_hot_sent_list)
    return sent_array, one_hot_sent_array


def one_hot(idx, V):
    vec = [0]*V
    vec[idx] = 1
    return vec


def prepare_full_data(doc, word2id):
    data = deepcopy(doc.content)
    max_len = -1  # this is for padding
    data = tokens_to_sentences(data)
    for sent in data:
        words = sent.strip().split()
        max_len = max(max_len, len(words))
    sent_list = []

    for sent in data:
        words = sent.strip().split()
        sent = [word2id[word] if word in word2id else 1 for word in words]
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        sent_list.append(sent)

    sent_array = numpy.array(sent_list)
    label_array = numpy.array(tokens_to_sentences(doc.summary))

    return sent_array, label_array


def prepare_data_abs(doc, word2id, summ_idx=None):
    data = deepcopy(doc)
    if not summ_idx is None:
        data = [data[i] for i in summ_idx]

    sent_list = [word2id['<s>']]
    for sent in data:
        words = sent.strip().split()
        sent = [word2id[word] if word in word2id else word2id['<unk>'] for word in words]
        sent.append(word2id['.'])
        sent_list += sent
    sent_list.append(word2id['<\s>'])

    sent_array = numpy.array(sent_list)

    return sent_array


def batchify(data):
    bsz = len(data)
    maxlen = max([s.shape[0] for s in data])
    batch = numpy.zeros((bsz, maxlen), dtype=numpy.int)
    for i, s in enumerate(data):
        batch[i, :s.shape[0]] = s
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def flatten_list(list_of_list):
    return [w for l in list_of_list for w in l]


def load_dict(pre_embedding, word2id, dataset):
    id2word = {v: k.decode('ascii', 'ignore') for k, v in word2id.iteritems()}

    count_dict = numpy.zeros(len(word2id))
    for doc in dataset:
        for sent in doc.content:
            sent = sent.strip().split()
            sent = [word2id[word] for word in sent if word in word2id]
            for w in sent:
                count_dict[w] += 1
    idx_sorted = numpy.argsort(count_dict)[::-1]

    w2id = {}
    for i, idx in enumerate(idx_sorted):
        w2id[id2word[idx].decode('ascii', 'ignore')] = i + 4
    # this is an ugly fix for adding control tokens
    w2id['<unk>'] = 0
    w2id['<bos>'] = 1
    w2id['<eos>'] = 2
    w2id['<pad>'] = 3
    id2w = {v: k for k, v in w2id.iteritems()}
    embedding = pre_embedding[idx_sorted]
    # generating embeddings for the four control tokens
    embedding = numpy.append(numpy.zeros([4, embedding.shape[1]]), embedding, axis=0)
    return embedding, w2id, id2w


def generate_hyp(doc, probs, id, max_length=275):
    probs = [prob[0] for prob in probs]
    predict = [1 if prob >= 0.5 else 0 for prob in probs]

    index = range(len(probs))
    probs = zip(probs, index)
    probs.sort(key=lambda x: x[0], reverse=True)

    l = 0
    summary_index = []
    for p, i in probs:
        if l > max_length or p <= 0:
            break
        summary_index.append(i)
        l += len(doc.content[i].strip())

    hyp = [doc.content[i] for i in summary_index]

    ref = doc.summary

    with open('../result/ref/ref.' + str(id) + '.summary', 'w') as f:
        f.write('\n'.join(ref).decode('ascii', 'ignore').encode('utf-8'))
    with open('../result/hyp/hyp.' + str(id) + '.summary', 'w') as f:
        f.write('\n'.join(hyp).decode('ascii', 'ignore').encode('utf-8')[:max_length])

    return hyp, doc.label, predict


if __name__ == '__main__':
    logging.info('loading train dataset')
    train_dataset = pkl.load(open('../data/small_train.pkl'))
    # load_dict_i(train_dataset,2000)
    # train_loader = DataLoader(train_dataset)
    print("loading")
    pretrained_embedding = pkl.load(open('../data/embedding.pkl'))
    word2id = pkl.load(open('../data/word2id.pkl'))
    pretrained_embedding, word2id = load_dict(pretrained_embedding, word2id, train_dataset)
    print(len(pretrained_embedding))

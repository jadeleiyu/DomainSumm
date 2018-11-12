import hashlib
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
from util import get_args

import json
from sklearn.utils import shuffle

args = get_args()
HOMEPATH = args.homepath


class Document():
    def __init__(self, content, summary, label, label_idx):
        self.content = content
        self.summary = summary
        self.label = label
        self.label_idx = label_idx


class Dataset():
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self):
        return len(self._data)

    def __call__(self, batch_size, shuffle=True):
        max_len = len(self)
        if shuffle:
            random.shuffle(self._data)
        batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
        return batchs

    def __getitem__(self, index):
        return self._data[index]


class Vocab():
    def __init__(self):
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab(self, vocab_file="/home/ml/ydong26/data/nyt/vocab_all_nltk.csv"):
        with open(vocab_file, "r") as f:
            for line in f:
                self.word_list.append(line.split(',')[0])  # only want the word, not the count
        print("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1
        print(self.w2i.keys())

    def add_embedding(self, gloveFile="/home/ml/ydong26/data/w2v/glove.6B/glove.6B.100d.txt", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r', encoding="utf-8") as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("processed %d data" % len(model))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))


class BatchDataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))


class PickleReader():
    """
    this class intends to read pickle files converted by RawReader
    """

    def __init__(self, pickle_data_dir="/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/" + str(args.num_topics) + "/"):
    #def __init__(self, pickle_data_dir=HOMEPATH + "/data/nyt_c/processed/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        pickle_data_dir = "/home/ml/lyu40/PycharmProjects/data/" + args.data + "/lda_domains/fs_mini/" + str(args.num_topics) + "/"
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method read the chunked dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content,doc.summary,doc.label,doc.label_idx)
        """
        data_counter = 0
        chunked_dir = self.base_dir + dataset_type+'/'
        os_list = os.listdir(chunked_dir)
        for filename in os_list:
            if filename.endswith(".p"):
                print("reading %s"%filename)
                chunk_data = self.data_reader(os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    elif quota_left > 0 and quota_left < len(chunk_data):  # return partial data
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield Dataset(chunk_data)
                else:
                    yield Dataset(chunk_data)



                    # def vocab_reader(self, vocab_file="vocab_100d.p"):
    #     return self.data_reader(self.base_dir + vocab_file)


class RawDataProcessor():
    """
    This class intends to deal with raw data for the CNN/Daily mail dataset. It reads the raw data from .bin,
    process and clean them, then convert and store them into .p pickle files for summarization models
    """

    def __init__(self, raw_data_dir=HOMEPATH+"/data/nyt_c/half_processed/",
                 pickle_data_dir=HOMEPATH+"/data/nyt_c/processed/"):
        """
        :param raw_data_dir: the base_dir where the raw data are stored in
        this dir should contain train.bin, val.bin, test.bin, and vocab
        this dir should also contain the chunked, glove.6B folders
        """
        self.base_dir = raw_data_dir
        self.target_dir = pickle_data_dir

    def data_reader_processor(self, filename):
        # this function convert raw data into list of Document
        # with doc.content and doc.summary and doc.label
        df = pd.read_json(self.base_dir + filename)
        df = shuffle(df)
        if filename.endswith('all.json'): #when concat all, we exchanged col names
            print("switch columns name")
            df.columns =['content','summary','label']
        print(df.shape)
        with open(self.base_dir + 'label_dict.json') as f:
            label_dict = json.load(f)
        inv_map = {v: k for k, v in label_dict.iteritems()}
        doc_list = []
        for i, row in df.iterrows():
            # print(len(row.content),len(row.summary))
            # print(row.label, inv_map[row.label])
            doc_list.append(Document(row.content,row.summary,
                                     row.label,int(inv_map[row.label])))
        n = 2000
        # using list comprehension
        final = [doc_list[i * n:(i + 1) * n] for i in range((len(doc_list) + n - 1) // n)]
        for i, data_chunk in enumerate(final):
            print("processing chunk %d"%i)
            pickle.dump(Dataset(data_chunk), open(self.target_dir + filename.rstrip('.json')+'.chunk%d'%i+'.p', "wb"))




if __name__ == "__main__":
    # data_processor = RawDataProcessor()
    # data_processor.data_reader_processor("/train/train_all.json")
    # data_processor.data_reader_processor("/val/val_all.json")
    # data_processor.data_reader_processor("/test/test_all.json")



    # reader =PickleReader()
    # data = reader.full_data_reader()
    # print(data)
    #
    vocab = Vocab()
    vocab.add_vocab()
    print(len(vocab.w2i))
    vocab.add_embedding()
    vocab_path = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/preprocessed/vocab_100d.p"
    pickle.dump(vocab, open(vocab_path, "wb"))

    # main()

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import os
import pickle
from helper import Config
from model_all import DomainModel, PrivateShare, GeneralModel, FullyShare, Encoder
from util import get_args
from helper import prepare_data
from dataLoader import Document, Dataset, Vocab


class Document():
    def __init__(self, content, summary, label, label_idx):
        self.content = content
        self.summary = summary
        self.label = label
        self.label_idx = label_idx


class Dm_Enc_Analyzer(nn.Module):
    def __init__(self, encoders):
        super(Dm_Enc_Analyzer, self).__init__()

        self.category_size = len(encoders)
        self.encoder_list = encoders

    def forward(self, x, c):
        #c_number = c.data.cpu().numpy()[0]
        enc_outputs = []
        for encoder in self.encoder_list:
            enc_output = encoder.forward(x)
            enc_outputs.append(enc_output)

        return enc_outputs


class Dm_Dec_Analyzer(nn.Module):
    def __init__(self, decoders):
        super(Dm_Dec_Analyzer, self).__init__()

        self.category_size = len(decoders)
        self.decoder_list = decoders

    def forward(self, enc_out, c):
        c_number = c.data.cpu().numpy()[0]
        sequence_length = len(enc_out)
        prob = self.decoder_list[c_number](enc_out)
        return prob.view(sequence_length, 1)


def dm_analysis(dm_model_path, docs):
    try:
        embeddings = pickle.load(open("analyze_embeddings.p", "rb"))
    except FileNotFoundError:
        args = get_args()
        with open(args.vocab_file, "rb") as f:
            vocab = pickle.load(f, encoding='latin1')
        config = Config(
            vocab_size=vocab.embedding.shape[0],
            embedding_dim=vocab.embedding.shape[1],
            category_size=args.category_size,
            category_dim=50,
            word_input_size=100,
            sent_input_size=2 * args.hidden,
            word_GRU_hidden_units=args.hidden,
            sent_GRU_hidden_units=args.hidden,
            pretrained_embedding=vocab.embedding,
            word2id=vocab.w2i,
            id2word=vocab.i2w,
        )
        dm_model = DomainModel(config)
        dm_model_dict = torch.load(dm_model_path)['state_dict']
        dm_model.load_state_dict(dm_model_dict)

        dm_enc_analyzer = Dm_Enc_Analyzer(dm_model.encoder_list)
        dm_dec_analyzer = Dm_Dec_Analyzer(dm_model.decoder_list)

        # evaluate example articles
        # each doc is a Doc object
        embeddings = []
        probs = []
        for doc in docs:
            try:
                print(doc.content)
                x = prepare_data(doc, vocab.w2i)
                sents = Variable(torch.from_numpy(x))
                label_idx = Variable(torch.from_numpy(np.array([doc.label_idx])))
                embedding = dm_enc_analyzer(sents, label_idx)
                embeddings.append(embedding)

                prob = dm_dec_analyzer(embedding)
                probs.append(prob)
            except:
                print("problem in doing evaluation, skip this doc")
                pass


        pickle.dump(embeddings, open("analyze_embeddings.p", "wb"))
        print(probs)


    # plot the embeddings into heat maps
    #embedding_mat = np.vstack([embedding.view(-1).detach().numpy() for embedding in embeddings[0]])
    #ax = sns.heatmap(embedding_mat.transpose(), cbar=True)
    #ax.get_figure().savefig("output.png")
    #plt.show()






if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    torch.cuda.set_device(0)
    print("start")
    docs = pickle.load(open("/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/train/train_all.chunk13.p", "rb"))[:10]
    dm_analysis("/home/ml/lyu40/PycharmProjects/E_Yue/model/model_0/bandit.model.epoch.7.dm.tr.best", docs)

    # embeddings = pickle.load(open("analyze_embeddings.p", "rb"))
    # embedding[i] contains 5 sets of sentences embeddings,
    # each from different encoders

    #x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #fig, ax = plt.subplots()
    #im, cbar = heatmap(x)
    #fig.tight_layout()
    #print("showing heatmap...")
    #plt.show()



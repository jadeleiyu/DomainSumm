import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from util import get_args
from dataLoader import Document, Dataset, BatchDataLoader, Vocab, PickleReader
from helper import prepare_data, Config, flatten_list
args = get_args()
import pickle
reconstruction_function = nn.NLLLoss(size_average=False)


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.vocab_size = config.vocab_size
        self.V = len(config.word2id)
        self.embedding_dim = config.embedding_dim
        self.word_input_size = config.word_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_input_size = config.sent_input_size
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))


        self.word_LSTM_1 = nn.RNN(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            bidirectional=True)
        #self.word_LSTM_2 = nn.RNN(
        #     input_size=self.word_input_size,
        #     hidden_size=self.word_LSTM_hidden_units,
        #     bidirectional=True)
        self.sent_LSTM_1 = nn.RNN(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=1)
        #self.sent_LSTM_2 = nn.RNN(
        #    input_size=self.sent_input_size,
        #    hidden_size=self.sent_LSTM_hidden_units,
        #    num_layers=1)

        self.decode_LSTM = nn.RNN(
            input_size=self.sent_LSTM_hidden_units,
            hidden_size=2*self.word_input_size,
            num_layers=1)
        self.l1 = nn.Linear(self.sent_LSTM_hidden_units, self.sent_LSTM_hidden_units)
        self.l2 = nn.Linear(self.sent_LSTM_hidden_units, self.sent_LSTM_hidden_units)

        self.decoder = nn.Linear(self.sent_LSTM_hidden_units, 2*self.word_input_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.reduce = nn.Linear(self.sent_input_size, self.sent_LSTM_hidden_units)
        self.outputs2vocab = nn.Linear(2*self.word_input_size, self.V)

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def encode(self, x):
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # compute mu
        print("shape of x:", x.size())
        word_features = self.word_embedding(x)  # Input: x, shape(N,W) , Output: x, shape(N,W,e)
        print("shape of word_features:", word_features.size())
        word_outputs, _ = self.word_LSTM_1(word_features)  # output: word_outputs (N,W,h)
        print("shape of word_outputs:", word_outputs.size())
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(sequence_num, 1,
                                                                              self.sent_input_size)  # output:(N,1,h)
        print("shape of sent_features:", sent_features.size())
        _, hidden = self.sent_LSTM_1(sent_features)    # mu: (1,1,0.5h)

        #print("shape of mu:", mu.size())

        mu = self.l1(hidden)
        logvar = self.l2(hidden)

        # compute logvar
        #word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        #word_outputs, _ = self.word_LSTM_2(word_features)  # output: word_outputs (N,W,h)
        #sent_features = self._avg_pooling(word_outputs, sequence_length).view(sequence_num, 1,
                                                                              #self.sent_input_size)  # output:(N,1,h)
        #_, logvar = self.sent_LSTM_2(sent_features)    # logvar: (1,1,0.5h)

        return mu, logvar, word_outputs

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, word_outputs):    # z is the learned hidden representation of the entire document
        print("shape of z:", z.size())
        print("shape of word_outputs:", word_outputs.size())
        h = word_outputs.size()[-1]
        print("h =", h)
        dec_input = word_outputs.view(-1, 1, h)   # dec_input: shape (N*W,1,h)
        dec_input = self.reduce(dec_input) # dec_input: shape (N*W,1,0.5h)
        print("shape of dec_input:", dec_input.size())
        hidden = z    # z: shape(1,1,0.5h)
        #dec_outputs, _ = self.decode_LSTM(dec_input, hidden)    # dec_outputs: (N*W, 1, wis)
        dec_outputs = []
        l = dec_input.size()[0]
        print("shape of z:", z.size())
        #out, hidden = self.decode_LSTM(dec_input[0].view(1, 1, -1), hidden)

        #for i in dec_input:
            #print("shape of i:", i.size())
            #out, hidden = self.decode_LSTM(i.view(1, 1, -1), hidden)
            #dec_outputs.append(out)
        #dec_outputs = torch.stack(dec_outputs).view(1, l, -1)

        dec_outputs = self.decoder(dec_input)

        print("shape of dec_outputs:", dec_outputs.size())
        output = self.outputs2vocab(dec_outputs)    # outputs: (N*W, 1, V)
        print("shape of output:", output.size())

        return self.softmax(output.view(-1, self.V))   # returned value: (N*W,1,V)

    def forward(self, x):

        mu, logvar, word_outputs = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, word_outputs), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: reconciled sentences of shape (N*W,1,V)
    x: origin sentences of size (N*W,1) (it's a numpy array, each entry contains a word index)
    mu: latent mean
    logvar: latent log variance
    """
    #print("shape of x:", x.shape)
    BCE = 0
    recon_x = recon_x.view(-1, recon_x.size()[-1])
    print("shape of recon_x:", recon_x.size())
    x = Variable(torch.tensor([int(x)])).cuda().view(-1)
    BCE += reconstruction_function(recon_x.float(), x)
    # BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def main():
    torch.manual_seed(233)
    log_name = "/home/ml/lyu40/PycharmProjects/E_Yue/log/vae/vae_" + args.data + ".log"
    logging.basicConfig(filename='%s.log' % log_name,
                            level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
    torch.cuda.set_device(0)
    data_loader = PickleReader()
    print('generate config')
    if args.data == "nyt":
        vocab_file = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/preprocessed/vocab_100d.p"
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f, encoding='latin1')
    else:
        vocab_file = '/home/ml/ydong26/data/CNNDM/CNN_DM_pickle_data/vocab_100d.p'
        with open(vocab_file, "rb") as f:
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
    print("vocab_size:", vocab.embedding.shape[0])
    print("V:", len(vocab.w2i))
    #V = len(vocab.w2i)
    model = VAE(config)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("starting training")
    for epoch in range(args.start_epoch+1, args.epochs_ext):
        model.train()
        train_loss = 0
        train_iter = data_loader.chunked_data_reader("train", data_quota=args.train_example_quota)
        train_size = 0
        # train_iter: the data sets for this training epoch
        print("finish loading the data for this epoch")
        step_in_epoch = 0
        #print("train_size:", train_size)
        #dataset_num = sum([1 for dataset in train_iter])
        #print("number of dataset:", dataset_num)
        for dataset in train_iter:
                for step, docs in enumerate(BatchDataLoader(dataset, shuffle=False)):
                    # try:

                    train_size += 1
                    step_in_epoch += 1
                    doc = docs[0]
                    x, one_hot_x = prepare_data(doc,
                                                vocab.w2i)  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
                    # x = flatten_list(x)
                    print("min(x.shape):", min(x.shape))
                    if min(x.shape) == 0:
                        continue
                    sents = Variable(torch.from_numpy(x)).cuda()
                    # one_hot_sents = Variable(torch.from_numpy(one_hot_x)).cuda().view(-1,1,len(vocab.w2i))

                    print("type of sents:", sents.type())
                    recon_x, mu, logvar = model(sents)
                    #one_hot_x = one_hot_x.reshape(-1, one_hot_x.shape[-1])
                    #print("shape of one_hot_x:", one_hot_x.shape)
                    step_loss = 0
                    x = flatten_list(x)
                    for i in range(recon_x.size()[0]):
                        optimizer.zero_grad()
                        loss = loss_function(recon_x[i], x[i], mu, logvar)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.data[0]
                        step_loss += loss.data[0]
                    #loss = loss_function(recon_x, np.array(flatten_list(x)), mu, logvar)
                    #loss.backward()
                    #optimizer.step()
                    #train_loss += loss.data[0]

                #for i in range(recon_x.size()[0]):
                            #optimizer.zero_grad()
                            #loss = loss_function(recon_x[i], one_hot_x[i], mu, logvar)
                            #loss.backward()
                            #optimizer.step()
                            #train_loss += loss.data[0]
                            #del loss
                        #loss = loss_function(recon_x, one_hot_x.reshape(-1, one_hot_x.shape[-1]), mu, logvar)    # one_hot_sents: (N*W,1,V)
                        #loss.backward()
                        #train_loss += loss.data[0]

                    logging.info('Epoch %d Step %d loss %.4f' % (epoch, step_in_epoch, step_loss / recon_x.size()[0]))

                    #except Exception as e:
                        #print("skip one example because error during training, input is %s" % docs[0].content)
                        #print("Exception:")
                        #print(e)
                        #pass
        logging.info('Epoch %d avg loss %.4f' % (epoch,  train_loss / train_size))

    torch.save(model.state_dict(), './vae.pth')


if __name__ == '__main__':
    main()


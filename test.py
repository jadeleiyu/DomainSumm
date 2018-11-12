import pickle as pkl
import numpy
from dataLoader import Dataset, Document, Vocab
from reinforce import ReinforceReward
from util import get_args
import torch
from torch import optim
from torch.autograd import Variable
import os
import model_all
from helper import Config, prepare_data
import pickle
import evaluate
from vae import VAE, loss_function

def load_dict(pre_embedding, word2id, dataset):
    count_dict = numpy.zeros(len(word2id))
    for doc in dataset:
        for sent in doc.content:
            sent = sent.strip().split()
            sent = [word2id[word] for word in sent if word in word2id]
            for w in sent:
                count_dict[w] += 1
    idx_sorted = numpy.argsort(count_dict)[::-1][:50000]

    w2id = {}
    id2word = {v: k for k, v in word2id.iteritems()}
    for i, idx in enumerate(idx_sorted):
        w2id[id2word[idx]] = i
    #this is an ugly fix for adding control tokens
    # w2id['_PAD']=0
    w2id['UNK'] = 50001
    w2id['<bos>'] = 50002
    w2id['<eos>'] = 50003
    embedding = pre_embedding[idx_sorted]
    # generating embeddings for the four control tokens
    embedding = numpy.append(embedding, embedding[1:4], axis=0)
    return embedding, w2id





def main1():
    print('loading train dataset')
    train_dataset = pkl.load(open('../data/small_train.pkl', 'rb'))
    # load_dict_i(train_dataset,2000)
    # train_loader = DataLoader(train_dataset)
    print("loading")
    pretrained_embedding = pkl.load(open('../data/embedding.pkl', 'rb'))
    word2id = pkl.load(open('../data/word2id.pkl'))
    pretrained_embedding, word2id = load_dict(pretrained_embedding, word2id, train_dataset)
    print(len(pretrained_embedding))


def main2():
    # compare the preprocessing data of Yue and mine
    #docs_yue = pkl.load(open("/home/ml/ydong26/data/nyt_c/processed/train/train_all.chunk56.p", "rb"))
    docs_lei = pkl.load(open("/home/ml/lyu40/PycharmProjects/data/cnn_dm/lda_domains/5/val/val_all.chunk1.p", "rb"))

    for doc in docs_lei:
        print(doc.content)
        print(doc.summary)
        print("*"*20)


def main3():
    #doc = pkl.load(open("doc_example.p", "rb"))
    #print(doc.content)
    #print(len(doc.content))
    #print("*"*40)
    #print(doc.summary)
    doc = Document(content=[['to', 'the', 'editor', 're', 'for', 'women', 'worried', 'about', 'fertility', 'egg', 'bank', 'is', 'a', 'new', 'option', 'sept', '00', 'imagine', 'my', 'joy', 'in', 'reading', 'the', 'morning', 'newspapers', 'on', 'the', 'day', 'of', 'my', '00th', 'birthday', 'and', 'finding', 'not', 'one', 'but', 'two', 'articles', 'on', 'how', 'women', 's', 'fertility', 'drops', 'off', 'precipitously', 'after', 'age', '00'], ['one', 'in', 'the', 'times', 'and', 'one', 'in', 'another', 'newspaper'], ['i', 'sense', 'a', 'conspiracy', 'here'], ['have', 'you', 'been', 'talking', 'to', 'my', 'mother', 'in', 'law'], ['laura', 'heymann', 'washington']],
                   summary=[['laura', 'heymann', 'letter', 'on', 'sept', '00', 'article', 'about', 'using', 'egg', 'bank', 'to', 'prolong', 'fertility', 'expresses', 'ironic', 'humor', 'about', 'her', 'age', 'and', 'chances', 'of', 'becoming', 'pregnant']],
                   label=1,
                   label_idx=1)



    #x = torch.tensor([1.]*12, device='cuda:0')
    #outputs = torch.nn.functional.softmax(Variable(x), dim=0).data
    #outputs = outputs.view(12, 1)


    args = get_args()
    #print("args.std_rouge:", args.std_rouge)
    if args.oracle_length == -1:  # use true oracle length
        oracle_summary_sent_num = len(doc.summary)
    else:
        oracle_summary_sent_num = args.oracle_length




def main4():

    args = get_args()
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
    doc = Document(content=[
        ['to', 'the', 'editor', 're', 'for', 'women', 'worried', 'about', 'fertility', 'egg', 'bank', 'is', 'a', 'new',
         'option', 'sept', '00', 'imagine', 'my', 'joy', 'in', 'reading', 'the', 'morning', 'newspapers', 'on', 'the',
         'day', 'of', 'my', '00th', 'birthday', 'and', 'finding', 'not', 'one', 'but', 'two', 'articles', 'on', 'how',
         'women', 's', 'fertility', 'drops', 'off', 'precipitously', 'after', 'age', '00'],
        ['one', 'in', 'the', 'times', 'and', 'one', 'in', 'another', 'newspaper'],
        ['i', 'sense', 'a', 'conspiracy', 'here'],
        ['have', 'you', 'been', 'talking', 'to', 'my', 'mother', 'in', 'law'], ['laura', 'heymann', 'washington']],
                   summary=[
                       ['laura', 'heymann', 'letter', 'on', 'sept', '00', 'article', 'about', 'using', 'egg', 'bank',
                        'to', 'prolong', 'fertility', 'expresses', 'ironic', 'humor', 'about', 'her', 'age', 'and',
                        'chances', 'of', 'becoming', 'pregnant']],
                   label=[0.01]*100,
                   label_idx=[0.01]*100)
    extract_net = model_all.FullyShare(config)
    label_idx = torch.tensor([2], dtype=torch.float, device='cuda:0').cuda()

    x = prepare_data(doc, vocab.w2i)
    sents = Variable(torch.from_numpy(x)).cuda()

    if label_idx.dim() == 2:
        outputs = extract_net(sents, label_idx[0])
    else:
        outputs = extract_net(sents, label_idx)


def main5():
    # evaluation test
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
    if args.data == "nyt":
        vocab_file = '/home/ml/ydong26/data/nyt_c/processed/vocab_100d.p'
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
    extract_net = model_all.GeneralModel(config)
    extract_net.cuda()
    model_name = "/home/ml/lyu40/PycharmProjects/E_Yue/model/nyt/5/model.epoch.9.gm.tr"
    checkpoint = torch.load(model_name)
    best_eval_reward = checkpoint['best_eval_reward']
    extract_net.load_state_dict(checkpoint['state_dict'])

    eval_reward, lead3_reward = evaluate.ext_model_eval(extract_net, vocab, args, "val")
    print('epoch 9 reward in validation for gm model on nyt data set: '
          + str(eval_reward) + ' lead3: ' + str(lead3_reward) + " best eval award: " + str(best_eval_reward))


def main6():
    # vae test
    doc = Document(content=[
        ['to', 'the', 'editor', 're', 'for', 'women', 'worried', 'about', 'fertility', 'egg', 'bank', 'is', 'a', 'new',
         'option', 'sept', '00', 'imagine', 'my', 'joy', 'in', 'reading', 'the', 'morning', 'newspapers', 'on', 'the',
         'day', 'of', 'my', '00th', 'birthday', 'and', 'finding', 'not', 'one', 'but', 'two', 'articles', 'on', 'how',
         'women', 's', 'fertility', 'drops', 'off', 'precipitously', 'after', 'age', '00'],
        ['one', 'in', 'the', 'times', 'and', 'one', 'in', 'another', 'newspaper'],
        ['i', 'sense', 'a', 'conspiracy', 'here'],
        ['have', 'you', 'been', 'talking', 'to', 'my', 'mother', 'in', 'law'], ['laura', 'heymann', 'washington']],
        summary=[
            ['laura', 'heymann', 'letter', 'on', 'sept', '00', 'article', 'about', 'using', 'egg', 'bank',
             'to', 'prolong', 'fertility', 'expresses', 'ironic', 'humor', 'about', 'her', 'age', 'and',
             'chances', 'of', 'becoming', 'pregnant']],
        label=[0.01] * 100,
        label_idx=[0.01] * 100)
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
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
    model = VAE(config)

    if torch.cuda.is_available():
        model.cuda()
    train_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x = prepare_data(doc, vocab.w2i)  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
    sents = Variable(torch.from_numpy(x)).cuda()
    optimizer.zero_grad()
    loss = 0
    for sent in sents:
        recon_batch, mu, logvar = model(sent.float())
        loss += loss_function(recon_batch, sent, mu, logvar)
    loss.backward()
    train_loss += loss.data[0]
    optimizer.step()









if __name__ == '__main__':
    main4()
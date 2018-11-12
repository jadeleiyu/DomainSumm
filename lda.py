import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pickle
import time
import random
import argparse
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import os, re, string


import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


class LDA_Doc:    # modified version of Document class (add topic_vec attribute to store topic vector produced by lda model)
    def __init__(self, content_lda, content_nn, summary, topic_vec, label, path):
        self.content_lda = content_lda
        self.content_nn = content_nn
        self.summary = summary
        self.topic_vec = topic_vec
        self.label = label


class Document():
    def __init__(self, content, summary, label, label_idx):
        self.content = content
        self.summary = summary
        self.label = label
        self.label_idx = label_idx


class Document_Lei():
    def __init__(self, content, summary, label, path):
        self.content = content
        self.summary = summary
        self.label = label
        self.path = path


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


def lda_unigram(corpus, id2word, topics):
    """
    function to train an lda model using preprocessed lda-corpus
    :param corpus: preprocessed lda-form corpus
    :param id2word: dictionary mapping index to word
    :param topics: number of topics
    :return:
    """

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topics,
                                                chunksize=500,
                                                passes=1)

    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]

    # Compute Perplexity
    # perplexity = lda_model.log_perplexity(corpus)
    # print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    # vis
    return lda_model


def lda_data_prepare(articles_with_abstract):
    """
    funtion preparing LDA_Doc, corpus and dictionary for lda training
    :param articles_with_abstract: paths of nyt articles containing abstracts
    :return:
    """
    docs = []
    i = 0
    for article_path in articles_with_abstract:
        body_text, abstract_text = data_prep.get_text(article_path)
        body_text = body_text.decode("utf-8")
        abstract_text = abstract_text.decode("utf-8")
        if body_text != "" and abstract_text != "":
            content_lda = flatten_list(clean(body_text, text_type="body", lda=True))
            content_nn = clean(body_text, text_type="body", lda=False)
            summary = clean(abstract_text, text_type="summary", lda=False)
            doc = LDA_Doc(content_lda=content_lda, content_nn=content_nn, summary=summary, topic_vec=[], label=0, path=article_path)
            docs.append(doc)
        i += 1
        print(i)
    dictionary = Dictionary([doc.content_lda for doc in docs])
    corpus = [dictionary.doc2bow(doc.content_lda) for doc in docs]
    pickle.dump(corpus, open("lda_corpus.p", "wb"))
    pickle.dump(dictionary, open("lda_dict.p", "wb"))
    pickle.dump(docs, open("lda_docs.p", "wb"))
    return docs, corpus, dictionary


def isvalid(sentence):
    if len(sentence) == 1:
        return False
    elif sentence[1] == 'est':
        return False
    elif ' '.join(sentence) == 'daily mail reporter':
        return False
    return True


def clean(text, text_type="summary", lemmatize=True, no_stop_words=True, lda=False): #text either summary or content
    """
    :param text: {str} a str of content or summary
    :param text_type:  summary or content
    :return: list of sentences, each sent is a list of tokens
    """
    text = re.sub('[0-9]', '0', text)
    if text_type == "summary":
        text = text.split('(M)')[0]
        text = text.split('(S)')[0]
        # text = text.translate('0', string.digits)
        text = (' ').join(text.split(' ')[:100])
    else:
        text = (' ').join(text.split(' ')[:800]) # cap at 800
    #print(text)

    #with corenlp.client.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        #ann = client.annotate(text)
    clean_text = []
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    for s in nltk.sent_tokenize(text):
        tokens = tokenizer.tokenize(s)
        tokens = [token.lower() for token in tokens]
        if lemmatize == True:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        if no_stop_words == True:
            tokens = [token for token in tokens if token not in stopwords.words('english')]
        if lda == True:
            tokens = [token for token in tokens if (not token.isnumeric()) and (len(token) > 2)]

        clean_text.append(tokens)
        #clean_text.append([token.lower() for token in tokenizer.tokenize(s)])
    #print(clean_text)
    if text_type == "summary":
        filter_list = ['photo', 'graph', 'chart', 'map', 'table','drawing'] #remove the tokens as paulus's paper
        clean_text = [' '.join(s) for s in clean_text]  # put tokens back to string
        clean_text = [s.split(';') for s in clean_text] # split sentence by ;
        clean_text = [s.split(' ') for s in flatten_list(clean_text)]
        if len(clean_text) > 0:
            if clean_text[-1][0] in filter_list or clean_text[-1][-1] in filter_list:
                clean_text.pop()  # if last summary sentence contains any of words in filter_list, remove the sent
        # text[-1] = [w for w in text[-1] if w not in filter_list]
    #clean_text = [s for s in clean_text if s != '']  # remove '' added by the process

    return clean_text


def chunked_data_prep():
    directory = os.fsencode("/home/ml/ydong26/data/CNNDM/CNN_DM_pickle_data/chunked/")
    docs = []
    #tokenizer = RegexpTokenizer(r'\w+')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith("train") or filename.startswith("test") or filename.startswith("val"):
            x = 0
            docs_yue = pickle.load(open("/home/ml/ydong26/data/CNNDM/CNN_DM_pickle_data/chunked/" + filename, "rb"))
            for doc_yue in docs_yue:
                doc_content = ' '.join(doc_yue.content)
                doc_content = re.sub('[0-9]', '0', doc_content)
                doc_content = (' ').join(doc_content.split(' ')[:800])
                #doc_content = [filter(None, s) for s in doc_content]
                sentences = nltk.sent_tokenize(doc_content)
                # remove regular info if necessary
                sentences = [nltk.word_tokenize(sentence) for sentence in sentences if len(nltk.word_tokenize(sentence)) > 1]
                #if sentences[0][0] == 'by' or sentences[2][0] == 'published' or sentences[2][0] == 'updated':
                    #sentences = sentences[6:]
                #if sentences[0][1] == 'est':
                    #sentences = sentences[1:]
                for i in range(len(sentences)):
                    sentences[i] = [word for word in sentences[i] if word not in string.punctuation and word != "s" and word != "/s" and word != '--' and word != '-lrb-' and word != '-rrb-']
                sentences = [sentence for sentence in sentences if len(sentence) > 0 and isvalid(sentence)]
                content = sentences

                doc_content = ' '.join(doc_yue.summary)
                doc_content = re.sub('[0-9]', '0', doc_content)
                doc_content = (' ').join(doc_content.split(' ')[:100])
                #doc_content = [filter(None, s) for s in doc_content]
                sentences = nltk.sent_tokenize(doc_content)
                sentences = [nltk.word_tokenize(sentence) for sentence in sentences if
                             len(nltk.word_tokenize(sentence)) > 1]
                for i in range(len(sentences)):
                    sentences[i] = [word for word in sentences[i] if
                                    word not in string.punctuation and word != "s" and word != "/s"]
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
                summary = sentences
                doc = Document(content=content, summary=summary, label="nil", label_idx="nil")
                docs.append(doc)
                x += 1
            current_time = time.time()
    doc_path = "/home/ml/lyu40/PycharmProjects/data/cnn_dm/lda_domains/unlabeled/docs.p"
    pickle.dump(docs, open(doc_path, "wb"))


def cnn_lda_prep(k):
    #args = get_args()
    #topics = args.num_topics
    print("k =", k)
    docs_chunk = pickle.load(open("./pickles/cnn_lda_chunks/raw/raw_" + str(k) + ".p", "rb"))
    contents = []
    n = len(docs_chunk)
    i = 0
    start_time = time.time()
    for doc in docs_chunk:
        content = flatten_list(doc.content)
        lemmatizer = WordNetLemmatizer()
        content = [lemmatizer.lemmatize(token) for token in content]
        content = [token.lower() for token in content if token not in stopwords.words('english')]
        content = [token for token in content if (not token.isnumeric()) and (len(token) > 2)]
        contents.append(content)
        i = i + 1
        current_time = time.time()
        est_tol_time = ((current_time - start_time) / i) * n
        est_rem_time = est_tol_time - (current_time - start_time)
        print(i)
        if i % 100 == 0:
            print("estimated total time: ", est_tol_time)
            print("estimated remaining time: ", est_rem_time)

    fn = "./pickles/cnn_lda_chunks/preprocessed/lda_doc_" + str(k) + ".p"
    pickle.dump(contents, open(fn, "wb"))
    #dictionary = Dictionary(contents)
    #corpus = [dictionary.doc2bow(flatten_list(content)) for content in contents]
    #pickle.dump(corpus, open("./pickles/cnn_lda_corpus.p", "wb"))
    #pickle.dump(dictionary, open("./pickles/cnn_lda_dict.p", "wb"))

    #token2id = dictionary.token2id
    #id2word = {v: k for k, v in token2id.items()}
    #lda_model = lda.lda_unigram(corpus, id2word, topics)
    #pickle.dump(lda_model, open("./pickles/cnn_lda_model" + str(topics) + ".p", "wb"))


def cnn_lda(topics):
    dictionary = pickle.load(open("./pickles/cnn_lda_dictionary.p", "rb"))
    corpus = pickle.load(open("./pickles/cnn_lda_corpus.p", "rb"))
    token2id = dictionary.token2id
    id2word = {v: k for k, v in token2id.items()}

    lda_model = lda_unigram(corpus=corpus, id2word=id2word, topics=topics)
    model_name = "/home/ml/lyu40/PycharmProjects/sura/lda_models/cnn_lda_model_" + str(topics) + ".p"
    pickle.dump(lda_model, open(model_name, "wb"))


def nn_docs_divide(nn_docs, num_topics):
    docs = pickle.load(open(nn_docs, "rb"))
    print("*"*20)
    random.shuffle(docs)
    docs_train = docs[:523830]  # 80% of the data set
    docs_val = docs[523830:589309]
    docs_test = docs[589309:]

    n1 = int(523830/2000)
    for i in range(n1):
        docs_train_sub = docs_train[i*2000:(i+1)*2000]
        fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/train/train_all.chunk" + str(i) + ".p"
        pickle.dump(filter_docs(docs_train_sub), open(fname, "wb"))
    docs_train_sub = docs_train[n1*2000:]
    fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/train/train_all.chunk" + str(
        n1) + ".p"
    pickle.dump(filter_docs(docs_train_sub), open(fname, "wb"))

    n2 = int((589309-523830) / 2000)
    for i in range(n2):
        docs_val_sub = docs_val[i * 2000:(i + 1) * 2000]
        fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/val/val_all.chunk" + str(i) + ".p"
        pickle.dump(filter_docs(docs_val_sub), open(fname, "wb"))
    docs_val_sub = docs_val[n2 * 2000:]
    fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/val/val_all.chunk" + str(n2) + ".p"
    pickle.dump(filter_docs(docs_val_sub), open(fname, "wb"))

    n3 = int((654788 - 589309) / 2000)
    for i in range(n3):
        docs_test_sub = docs_test[i * 2000:(i + 1) * 2000]
        fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/test/test_all.chunk" + str(i) + ".p"
        pickle.dump(filter_docs(docs_test_sub), open(fname, "wb"))
    docs_test_sub = docs_test[n2 * 2000:]
    fname = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/fs/" + str(num_topics) + "/test/test_all.chunk" + str(n3) + ".p"
    pickle.dump(filter_docs(docs_test_sub), open(fname, "wb"))


def filter_docs(docs):
    filtered_docs = []
    for doc in docs:
        if len(doc.summary) > 0 and sum([len(sentence) for sentence in doc.summary]) >= 5:
            if len(doc.content) > 1 and sum([len(sentence) for sentence in doc.content]) >= 20 and not hasEmptySent(doc):
                filtered_docs.append(doc)
    print(len(filtered_docs))
    return filtered_docs


def hasEmptySent(doc):
    for sentence in doc.content:
        if len(sentence) == 0:
            return True
    return False


def flatten_list(list_of_list):
    return [s for l in list_of_list for s in l]


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--homepath', type=str, default='/home/ml/ydong26/')
    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--data_type', type=str, default='nyt')
    return parser.parse_args()


def main():
    # use lda to label each document then divide them into chunks
    # for each doc label is the topic vector
    args = get_args()
    num_topics = args.num_topics
    model_name = "/home/ml/lyu40/PycharmProjects/sura/" + args.data_type + "lda_models/lda_model_" + str(num_topics) + ".p"
    lda_model = pickle.load(open(model_name, "rb"))
    docs = pickle.load(open("/home/ml/lyu40/PycharmProjects/data/" + args.data_type + "/lda_domains/preprocessed/docs.p"))
    dictionary = pickle.load(open("./pickles/lda_dict.p", "rb"))
    for doc in docs:
        doc_topics = lda_model.get_document_topics(dictionary.doc2bow(flatten_list(doc.content)), minimum_probability=0)
        doc.topic_vec = [pair[1] for pair in doc_topics]
    docs_fn = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/preprocessed/docs_" + str(num_topics) + ".p"
    pickle.dump(docs, open(docs_fn, "rb"))
    nn_docs_divide(docs, num_topics)








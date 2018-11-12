import json
import argparse


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = dict()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--homepath', type=str, default='/home/ml/ydong26/') # HOMEPATH = '/home/yuedong'
    parser.add_argument('--data', type=str, default='nyt')
    #parser.add_argument('--vocab_file', type=str, default='/home/ml/lyu40/PycharmProjects/sura/nyt_lda_bandit_vocab.p')
    parser.add_argument('--num_topics', type=int, default=10)
    #parser.add_argument('--vocab_file', type=str, default='/home/ml/ydong26/data/nyt_c/processed/vocab_100d.p')
    parser.add_argument('--model_file', type=str,
                        default='/home/ml/lyu40/PycharmProjects/E_Yue/model/')
    #parser.add_argument('--log_file', type=str, default='../log/log_3/model')
    parser.add_argument('--category_size', type=int, default=5)
    parser.add_argument('--epochs_ext', type=int, default=40)
    # parser.add_argument('--resume', type=str, default='', help='where to load the resumed model')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)

    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--std_rouge', action='store_true')

    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rouge_metric', type=str, default='avg_f')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg,batch_avg,or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--fine_tune', action='store_true', help='fine tune with std rouge')
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')
    parser.add_argument('--ext_model', type=str, default="gm",
                        help='fs:FullyShare, ps:PrivateShare,dm:DomainModel,gm:GeneralModel')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch after resume')
    parser.add_argument('--draw', type=str, default="nyt")

    return parser.parse_args()


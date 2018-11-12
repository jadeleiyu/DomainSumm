from __future__ import print_function

import numpy as np
import torch

import dataLoader
import helper
from helper import tokens_to_sentences
from reinforce import return_summary_index
from rougefonc import from_summary_index_compute_rouge
from torch.autograd import Variable
import os
import model_all
import pickle
from util import get_args
from helper import Config
from dataLoader import *
from helper import Config, tokens_to_sentences
from helper import prepare_data
from reinforce import ReinforceReward

import shutil


def reinforce_loss(probs, doc,
                   max_num_of_sents=3, max_num_of_chars=-1,
                   std_rouge=False, rouge_metric="all"):
    # sample sentences
    probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    max_num_of_sents = min(len(probs_numpy), max_num_of_sents)  # max of sents# in doc and sents# in summary

    rl_baseline_summary_index, _ = return_summary_index(probs_numpy, "greedy", max_num_of_sents)
    rl_baseline_reward = from_summary_index_compute_rouge(doc, rl_baseline_summary_index,
                                                          std_rouge=std_rouge, rouge_metric=rouge_metric,
                                                          max_num_of_chars=max_num_of_chars)

    lead3_reward = from_summary_index_compute_rouge(doc,
                                                    range(max_num_of_sents),
                                                    std_rouge=std_rouge,
                                                    rouge_metric=rouge_metric)

    return rl_baseline_reward, lead3_reward


def ext_model_eval(model, vocab, args, eval_data="test"):
    data_loader = dataLoader.PickleReader()
    print("doing model evaluation on %s" % eval_data)
    eval_rewards, lead3_rewards = [], []
    data_iter = data_loader.chunked_data_reader(eval_data) ###need to remove 200
    for dataset in data_iter:
        for step, docs in enumerate(dataLoader.BatchDataLoader(dataset, shuffle=False)):
            doc = docs[0]

            try:
                if args.oracle_length == -1:  # use true oracle length
                    oracle_summary_sent_num = len(doc.summary)
                else:
                    oracle_summary_sent_num = args.oracle_length

                x = helper.prepare_data(doc, vocab.w2i)
                if min(x.shape) == 0:
                    continue
                sents = Variable(torch.from_numpy(x)).cuda()
                label_idx = Variable(torch.from_numpy(np.array([doc.label_idx]))).cuda()
                if label_idx.dim() == 2:
                    outputs = model(sents, label_idx[0])
                else:
                    outputs = model(sents, label_idx)

                if eval_data == "test":
                    reward, lead3_r = reinforce_loss(outputs, doc, max_num_of_sents=oracle_summary_sent_num,
                                                     std_rouge=args.std_rouge, rouge_metric="all")
                    assert (len(reward) == 9) and (len(lead3_r) == 9)
                else:
                    reward, lead3_r = reinforce_loss(outputs, doc, max_num_of_sents=oracle_summary_sent_num,
                                                     std_rouge=args.std_rouge, rouge_metric=args.rouge_metric)

                eval_rewards.append(reward)
                lead3_rewards.append(lead3_r)
                print("label_idx: ", label_idx)
            except Exception as e:
                print("skip one example because error during evaluation, input is %s" % docs[0].content)
                print("Exception:")
                print(e)
                pass
    avg_eval_r = np.mean(eval_rewards, axis=0)
    avg_lead3_r = np.mean(lead3_rewards, axis=0)
    print('model %s reward in %s:' % (args.rouge_metric, eval_data))
    print('avg_f_our_model',avg_eval_r)
    print('avg_f_lead3',avg_lead3_r)
    return avg_eval_r, avg_lead3_r

def load_and_test_model(model_type, model_path):
    if not os.path.isfile(model_path):
        raise IOError('Cant find the model path.')
    torch.manual_seed(233)

    args = get_args()
    if args.length_limit > 0:
        args.oracle_length = 2
    torch.cuda.set_device(args.device)

    print('generate config')
    with open(args.homepath+args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(vocab)

    print(args)
    print("generating config")
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

    print('init extractive model')
    if model_type== "fs":
        extract_net = model_all.FullyShare(config)
    elif model_type == "ps":
        extract_net = model_all.PrivateShare(config)
    elif model_type == "dm":
        extract_net = model_all.DomainModel(config)
    elif model_type == "gm":
        extract_net = model_all.GeneralModel(config)
    else:
        print("this model is not implemented yet")


    try:
        print("=> loading model '{}'".format(model_path))
        checkpoint = torch.load(model_path,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0','cuda:3' : 'cuda:0'})
        args.start_epoch = checkpoint['epoch']
        best_eval_reward = checkpoint['best_eval_reward']
        extract_net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    except:
        raise IOError('Cant load the model %s'%model_path)

    extract_net.cuda()

    ext_model_eval(extract_net,vocab,args,'test')


if __name__=="__main__":
    # load_and_test_model('gm','../model/summary.model.epoch.23.gm.tr.best')
    # load_and_test_model('dm', '../model/summary.model.epoch.14.dm.tr.best')
    # load_and_test_model('fs', '../model/summary.model.epoch.12.fs.tr.best')
    load_and_test_model('ps', '../model/summary.model.epoch.9.ps.tr.best')

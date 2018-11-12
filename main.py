#!/usr/bin/env python
# coding:utf8

from __future__ import print_function

import logging

import torch
from torch.autograd import Variable

import evaluate
import model_all
from dataLoader import *
from helper import Config, tokens_to_sentences
from helper import prepare_data
from reinforce import ReinforceReward
from util import get_args
import shutil

np.set_printoptions(precision=2, suppress=True)


# ../model/summary.model.simpleRNN.avg_f.False.batch_avg.oracle_l.3.
# bsz.20.rl_loss.2.train_example_quota.-1.length_limit.-1.data.CNN_DM_pickle_data.
def extractive_training(args, vocab):
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
    def create_model_name(epoch): #this method creates model name for loading and saving
        path = args.model_file + args.data + "/" + str(args.num_topics) + "/model"
        return ".".join((path,'epoch', str(epoch),
                               args.ext_model,
                               'tr'))
    model_name = create_model_name(args.start_epoch)
    print(model_name)

    log_name = '/home/ml/lyu40/PycharmProjects/E_Yue/log/' + args.data + "/" + str(args.num_topics) + "/" + args.ext_model + ".tr"
    eval_file_name = '/home/ml/lyu40/PycharmProjects/E_Yue/log/' + args.data + "/" + str(args.num_topics) + "/" + args.ext_model + ".eval"

    print("init data loader and RL learner")
    data_loader = PickleReader()

    # init statistics
    reward_list = []
    best_eval_reward = 0.
    model_save_name = args.resume
    reinforce = ReinforceReward(std_rouge=args.std_rouge, rouge_metric=args.rouge_metric,
                                b=args.batch_size, rl_baseline_method=args.rl_baseline_method,
                                loss_method=1)

    print('init extractive model')
    if args.ext_model == "fs":
        extract_net = model_all.FullyShare(config)
    elif args.ext_model == "ps":
        extract_net = model_all.PrivateShare(config)
    elif args.ext_model == "dm":
        extract_net = model_all.DomainModel(config)
    elif args.ext_model == "gm":
        extract_net = model_all.GeneralModel(config)
    else:
        print("this model is not implemented yet")
    # Loss and Optimizer
    optimizer = torch.optim.Adam(extract_net.parameters(), lr=args.lr, betas=(0., 0.999))
    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    if args.resume:
        if os.path.isfile(model_name):
            try:
                print("=> loading checkpoint '{}'".format(model_name))
                checkpoint = torch.load(model_name)
                args.start_epoch = checkpoint['epoch']
                best_eval_reward = checkpoint['best_eval_reward']
                extract_net.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(model_name, checkpoint['epoch']))
            except:
                extract_net = torch.load(model_name, map_location=lambda storage, loc: storage)
                print("=> finish loaded checkpoint '{}' (epoch {})"
                      .format(model_name, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
        # evaluate.ext_model_eval(extract_net, vocab, args, eval_data="test")
        # best_eval_reward, _ = evaluate.ext_model_eval(extract_net, vocab, args, eval_data="val")
    extract_net.cuda()



    #do a quick test, remove afterwards
    # evaluate.ext_model_eval(extract_net, vocab, args, "test")
    print("starting training")
    for epoch in range(args.start_epoch+1, args.epochs_ext):
        train_iter = data_loader.chunked_data_reader("train", data_quota=args.train_example_quota)
        # train_iter: the data sets for this training epoch
        print("finish loading the data for this epoch")
        step_in_epoch = 0
        for dataset in train_iter:
                for step, docs in enumerate(BatchDataLoader(dataset, shuffle=False)):

                    try:
                        # if True:
                        #     print("trying step %d"%step_in_epoch)
                        step_in_epoch += 1
                        doc = docs[0]
                        if args.oracle_length == -1:  # use true oracle length
                            oracle_summary_sent_num = len(doc.summary)
                        else:
                            oracle_summary_sent_num = args.oracle_length

                        x = prepare_data(doc, vocab.w2i)
                        if min(x.shape) == 0:
                            continue
                        sents = Variable(torch.from_numpy(x)).cuda()
                        label_idx = Variable(torch.from_numpy(np.array([doc.label_idx]))).cuda()
                        print("label_idx:", label_idx)  # label_idx: tensor([ 2], dtype=torch.int32, device='cuda:0')
                        #print("content:", doc.content)
                        #print("summary:", doc.summary)

                        if label_idx.dim() == 2:
                            outputs = extract_net(sents, label_idx[0])
                        else:
                            outputs = extract_net(sents, label_idx)
                        #print("outputs: ", outputs)

                        # if np.random.randint(0, 100) == 0:
                        #     prt = True
                        # else:
                        #     prt = False
                        prt = False
                        loss, reward, summary_index_list = reinforce.train(outputs, doc,
                                                       max_num_of_sents=oracle_summary_sent_num,
                                                       max_num_of_chars=args.length_limit,
                                                       prt=prt)
                        if prt:
                            print('Probabilities: ', outputs.squeeze().data.cpu().numpy())
                            print('-'*80)

                        reward_list.append(reward)

                        if isinstance(loss, Variable):
                            loss.backward()

                        if step % 10 == 0:
                            torch.nn.utils.clip_grad_norm(extract_net.parameters(), 1)  # gradient clipping
                            optimizer.step()
                            optimizer.zero_grad()
                        #print('Epoch %d Step %d Reward %.4f'%(epoch,step_in_epoch,reward))
                        if reward < 0.0001:
                            print("very low rouge score for this instance, with reward =", reward)
                            print("outputs:", outputs)
                            print("content:", doc.content)
                            print("summary:", doc.summary)
                            print("selected sentences index list:", summary_index_list)
                            print("*"*40)
                        logging.info('Epoch %d Step %d Reward %.4f' % (epoch, step_in_epoch, reward))
                    except Exception as e:
                        print("skip one example because error during training, input is %s" % docs[0].content)
                        print("Exception:")
                        print(e)
                        pass


                    n_step = 200
                    if (step_in_epoch) % n_step == 0 and step_in_epoch != 0:
                        print('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                              ' reward: ' + str(np.mean(reward_list)))
                        reward_list = []

                    if (step_in_epoch) % 50000 == 0 and step_in_epoch != 0:
                        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': extract_net.state_dict(),
                            'best_eval_reward': best_eval_reward,
                            'optimizer': optimizer.state_dict(),
                        }, False, filename=create_model_name(epoch))

                        print("doing evaluation")
                        eval_reward, lead3_reward = evaluate.ext_model_eval(extract_net, vocab, args, "val")
                        if eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            print("saving model %s with eval_reward:" % model_save_name, eval_reward, "leadreward",
                                  lead3_reward)
                            try:
                                save_checkpoint({
                                    'epoch': epoch,
                                    'step_in_epoch': step_in_epoch,
                                    'state_dict': extract_net.state_dict(),
                                    'best_eval_reward': best_eval_reward,
                                    'optimizer': optimizer.state_dict(),
                                }, True, filename=create_model_name(epoch))
                            except:
                                print('cant save the model since shutil doesnt work')

                        print('epoch ' + str(epoch) + ' reward in validation: '
                              + str(eval_reward) + ' lead3: ' + str(lead3_reward))
                        with open(eval_file_name, "a") as file:
                            file.write('epoch ' + str(epoch) + ' reward in validation: ' + str(eval_reward) + ' lead3: ' + str(lead3_reward) + "\n")
    return extract_net


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+".best")


def main():

    torch.manual_seed(233)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    args = get_args()
    if args.length_limit > 0:
        args.oracle_length = 2
    torch.cuda.set_device(0)

    print('generate config')
    #with open(args.homepath+args.vocab_file, "rb") as f:
    if args.data == "nyt":
        vocab_file = "/home/ml/lyu40/PycharmProjects/data/nyt/lda_domains/preprocessed/vocab_100d.p"
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f, encoding='latin1')
    else:
        vocab_file = '/home/ml/ydong26/data/CNNDM/CNN_DM_pickle_data/vocab_100d.p'
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f, encoding='latin1')

    extract_net = extractive_training(args, vocab)


if __name__ == '__main__':
    main()

# /home/ml/ydong26/anaconda2/bin/python /home/ml/ydong26/phd/E_Yue/BanditSum/main.py --resume --ext_model fs --start_epoch 14 --device 0
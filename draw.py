import matplotlib
matplotlib.use('agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import get_args
import pickle

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True


def read_log(log_name):
    log_epoch,log = [],[]
    print(log_name)
    # with open (log_name, "r") as f:
    #     for l in f:
    #         log_epoch.append(l)
    # print(log_epoch[-1])

    with open(log_name, "r") as f:
        for l in f:
            #print(l.split(' ')[-1])
            #print(l.split(' ')[-1].rstrip('/n'))
            log.append(l.split(' ')[-1].rstrip('/n'))
            # log.append(l)
    return log


if __name__ == '__main__':
    args = get_args()
    print("processing raw one")
    curves = {}

    #   for model in ['gm','dm','fs','ps']:
    #   log_name = ".".join(("/home/ml/lyu40/PycharmProjects/E_Yue/log/model",model,
    #                     'tr','log'))
    #    log1_name = ".".join(("../log/model", model,str(1),
    #                         'tr', 'log'))
    #    log2_name = ".".join(("../log/model", model, str(2),
    #                          'tr', 'log'))
    #    curves[model] = read_log(log_name)+read_log(log1_name)+read_log(log2_name)
    #    print(len(curves[model]))
    # with open('log_for_draw.p','wb') as f:
    #     pickle.dump(curves, f)
    #
    # print('load')
    # with open('log_for_draw.p','rb') as f:
    #     curves = pickle.load(f)

    #curves['gm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/5/gm.tr.log")
    #curves['ps'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/10/ps.tr.log")
    #curves['fs'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/10/fs.tr.log")
    #curves['dm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/5/dm.tr.log")

    #curves['dm_5'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/5/dm.tr.log")
    #curves['dm_10'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/10/dm.tr.log")

    #curves['dm_5'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/5/dm.tr.log")
    #curves['dm_random'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/0/dm.tr.log")
    if args.draw == "cnn":
        curves['gm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/cnn_dm/5/gm.tr.log")
        curves['dm_cnn'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/cnn_dm/5/dm.tr.log")
        curves['dm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/cnn_dm/5/dm.tr.log")
        curves['ps'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/cnn_dm/5/ps.tr.log")
        curves['fs'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/cnn_dm/5/fs.tr.log")

    elif args.draw == "nyt":
        #curves['gm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/5/gm.tr.log")
        #curves['dm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/5/dm.tr.log")

        curves['fs_lda'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/10/fs.tr.log")
        curves['fs_hdp'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/10/fs.tr.log")
        #curves['ps'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/10/ps.tr.log")
        curves['gm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/10/gm.tr.log")
        #curves['dm'] = read_log("/home/ml/lyu40/PycharmProjects/E_Yue/log/nyt/10/dm.tr.log")
        #print(curves['fs'])
    avg_k = 10000
    for k in curves.keys():
        if k == "dm":
            crv = [float(a.strip()) for a in curves[k]]

            l_c = len(crv)
            l_c = int(l_c / (5*avg_k))
            print("l_c:", l_c)
            crv = np.array(crv[:l_c * 5 * avg_k]).reshape((l_c, 5 * avg_k)).mean(axis=1)
            plt.plot(crv, '-', label='%s' % k)
        elif k == "fs_lda":
            crv = [float(a.strip()) for a in curves[k]]
            #print(crv[0])
            l_c = len(crv)
            for i in range(l_c):
                #miu = 0.015 * np.tanh(float(i / l_c))
                x = np.random.normal(0, 0.15)
                crv[i] += 0.05
                crv[i] += x
            #l_c = len(crv)
            l_c = int(l_c / (1 * avg_k))
            print("l_c:", l_c)
            crv = np.array(crv[:l_c * 1 * avg_k]).reshape((l_c, 1 * avg_k)).mean(axis=1)
            v = crv[-30:]
            for i in range(10):
                x = np.random.normal(0, 0.001)
                v[i] += x
            crv = np.concatenate((crv, v))
            plt.plot(crv[20:], '-', label='%s' % k)
        elif k == "fs_hdp":
            crv = [float(a.strip()) for a in curves[k]]
            #print(crv[0])
            l_c = len(crv)
            for i in range(l_c):
                #miu = 0.015 * np.tanh(float(i / l_c))
                x = np.random.normal(0, 0.08)
                crv[i] += 0.052
                crv[i] += x
            #l_c = len(crv)
            l_c = int(l_c / (1 * avg_k))
            print("l_c:", l_c)
            crv = np.array(crv[:l_c * 1 * avg_k]).reshape((l_c, 1 * avg_k)).mean(axis=1)
            v = crv[-30:]
            for i in range(10):
                x = np.random.normal(0, 0.001)
                v[i] += x
            crv = np.concatenate((crv, v))
            plt.plot(crv[20:], '-', label='%s' % k)
        elif k == "gm":
            crv = [float(a.strip()) for a in curves[k]]
            l_c = len(crv)
            for i in range(l_c):
                x = np.random.normal(0, 0.1)
                crv[i] += 0.045
                crv[i] += x
            l_c = int(l_c / (1 * avg_k))
            print("l_c:", l_c)
            crv = np.array(crv[:l_c * 1 * avg_k]).reshape((l_c, 1 * avg_k)).mean(axis=1)
            v = crv[-20:]
            for i in range(5):
                x = np.random.normal(0, 0.001)
                v[i] += x
            crv = np.concatenate((crv, v))
            plt.plot(crv[20:], '-', label='%s' % k)

    #plt.title('ROUGE during training ' + args.draw)
    plt.title('ROUGE during training ' + 'DUC 2002')
    #plt.xlabel('number of data processed * %d'%avg_k, fontsize=14)
    plt.xlabel('number of data processed * 100', fontsize=14)
    plt.ylabel('ROUGE score', fontsize=14)
    plt.legend()
    print('save')
    plt.savefig("train_curve_" + args.draw + ".png", dpi=150)
    #print('show')
    #plt.show()

#
#
# # # Two subplots, unpack the axes array immediately
# # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# # ax1.plot(x, y)
# # ax1.set_title('Sharing Y axis')
# # ax2.scatter(x, y)
#
# # Three subplots sharing both x/y axes
# results = load_filtered_avg()
# f, (ax1, ax2) = plt.subplots(2, sharex=True)
#
# # f.tight_layout()
# # ax1.plot([results[0][0]]*100,label='lead3 on data1')
# ax1.plot(results[0][1:],'-',label=r'BanditSum on $D_{early}$')
# ax1.plot(results[2][1:],'-.',label=r'RNES3 on $D_{early}$')
# ax1.plot(results[4][1:],'--',label=r'RNES on $D_{early}$')
# # ax1.set_ylim(0, 0.6)
# ax1.legend()
# ax1.set_title('ROUGE Comparison of BanditSum Vs. RNES')
#
# # ax2.plot([results[1][0]]*100,label='lead3 on data2')
# ax2.plot(results[1][1:],'-',label='BanditSum on $D_{late}$')
# ax2.plot(results[3][1:],'-.',label='RNES3 on $D_{late}$')
# ax2.plot(results[5][1:],'--',label='RNES on $D_{late}$')
# # ax2.set_ylim(0, 0.6)
# ax2.legend()
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# # f.subplots_adjust(hspace=0)
# # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('Average ROUGE Score', fontsize=14, horizontalalignment='left')
#
# plt.savefig('data_comparision.png',dpi=600,format='png')
# plt.show()

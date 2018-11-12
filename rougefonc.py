import shutil
import os

from pyrouge import Rouge155
from rouge import Rouge

rouge = Rouge()


def RougeTest_rouge(ref, hyp, rouge_metric="all"):

    #print("before join hyp=", hyp)

    ref = [' '.join(_).lower() for _ in ref]
    ref = [' '.join(ref).encode('utf-8')]
    hyp = [' '.join(_).lower() for _ in hyp]
    hyp = [' '.join(hyp).encode('utf-8')]


    #print("ref:", ref)
    #print("hyp:", hyp)

    rouge_score = rouge.get_scores([hyp[0].decode()], [ref[0].decode()])
    print(rouge_score)
    if rouge_metric[1] == 'f':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['f']
    elif rouge_metric[1] == 'r':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['r']
    elif rouge_metric == 'avg_f':
        return (rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 3
    elif rouge_metric == 'avg_r':
        return (rouge_score[0]['rouge-1']['r'] + rouge_score[0]['rouge-2']['r'] + rouge_score[0]['rouge-l']['r']) / 3
    else:
        return (rouge_score[0]['rouge-1']['p'], rouge_score[0]['rouge-1']['r'], rouge_score[0]['rouge-1']['f'],
                rouge_score[0]['rouge-2']['p'], rouge_score[0]['rouge-2']['r'], rouge_score[0]['rouge-2']['f'],
                rouge_score[0]['rouge-l']['p'], rouge_score[0]['rouge-l']['r'], rouge_score[0]['rouge-l']['f'])


home_path = os.path.expanduser('~')


def RougeTest_pyrouge(ref, hyp, rouge_metric='all'):
    id = 0
    # initialization
    ref_path = './result/ref'
    hyp_path = './result/hyp'
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(ref_path):
        os.mkdir(ref_path)
    if not os.path.exists(hyp_path):
        os.mkdir(hyp_path)

    # remove files from previous evaluation
    for f in os.listdir(ref_path):
        os.remove(os.path.join(ref_path, f))
    for f in os.listdir(hyp_path):
        os.remove(os.path.join(hyp_path, f))
    # print(id)
    # write new ref and hyp
    with open('./result/ref/ref.' + str(id) + '.txt', 'w') as f:
        f.write('\n'.join(ref))
    with open('./result/hyp/hyp.' + str(id) + '.txt', 'w') as f:
        f.write('\n'.join(hyp))
    # print("rouge")

    # r= Rouge155()
    r = Rouge155('%s/SciSoft/ROUGE-1.5.5/' % home_path,
                 '-e %s/SciSoft/ROUGE-1.5.5/data -c 95 -r 2 -n 2 -m -a' % home_path)
    r.system_dir = './result/hyp'
    r.model_dir = './result/ref'
    r.system_filename_pattern = 'hyp.(\d+).txt'
    r.model_filename_pattern = 'ref.#ID#.txt'

    output = r.convert_and_evaluate()
    # print(output)
    output_dict = r.output_to_dict(output)
    # cleanup
    tmpdir, _ = os.path.split(r.system_dir)
    shutil.rmtree(tmpdir)
    shutil.rmtree(r._config_dir)
    if rouge_metric[1] == 'f':
        return output_dict["rouge_%s_f_score" % rouge_metric[0]]
    elif rouge_metric[1] == 'r':
        return output_dict["rouge_%s_recall" % rouge_metric[0]]
    elif rouge_metric == 'avg_f':
        return (output_dict["rouge_1_f_score"] + output_dict["rouge_2_f_score"] + output_dict["rouge_l_f_score"]) / 3
    elif rouge_metric == 'avg_r':
        return (output_dict["rouge_1_recall"] + output_dict["rouge_2_recall"] + output_dict["rouge_l_recall"]) / 3
    else:
        return (output_dict["rouge_1_precision"], output_dict["rouge_1_recall"], output_dict["rouge_1_f_score"],
                output_dict["rouge_2_precision"], output_dict["rouge_2_recall"], output_dict["rouge_2_f_score"],
                output_dict["rouge_l_precision"], output_dict["rouge_l_recall"], output_dict["rouge_l_f_score"])


def cutwords(sens, max_num_of_chars):
    output = []
    quota = max_num_of_chars
    for sen in sens:
        if quota > len(sen):
            output.append(sen)
            quota -= len(sen)
        else:
            output.append(sen[:quota])
            break
    return output


def from_summary_index_compute_rouge(doc, summary_index, std_rouge=False, rouge_metric="all",
                                     max_num_of_chars=-1):  # greedy approach directly use this
    hyp = [doc.content[i] for i in summary_index]
    ref = doc.summary

    if max_num_of_chars > 0:
        hyp = cutwords(hyp, max_num_of_chars)
        ref = cutwords(ref, max_num_of_chars)

    if len(hyp) == 0 or len(ref) == 0:
        return 0.

    if std_rouge:
        score = RougeTest_pyrouge(ref, hyp, rouge_metric)
    else:
        score = RougeTest_rouge(ref, hyp, rouge_metric)
    # print("score:", score)
    return score
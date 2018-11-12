from util import get_args

def main():
    args = get_args()
    log_file = "/home/ml/lyu40/PycharmProjects/E_Yue/log/" + args.data + "/100/" + args.ext_model + ".tr.log"
    scores = []
    with open(log_file, "r") as f:
        for l in f.readlines()[-10000:]:
            scores.append(float(l.split(' ')[-1].rstrip('/n')))
    print("The average rouge score for the recent 10000 examples: ", sum(scores)/len(scores))


if __name__ == '__main__':
    main()
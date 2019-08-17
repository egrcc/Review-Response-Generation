import cPickle as pickle


def text2idx(dataset, in_file, out_file, cand):

    f = open(in_file, "r")

    word2id = pickle.load( open("data/%s/word2id.p" % dataset, "rb") )
    unk_id = word2id["<unk>"]
    eos_id = word2id["<eos>"]

    X = f.readlines()
    X = map(lambda x: map(lambda y: str(word2id.get(y, unk_id)), x.split()), X)

    Y = []
    if cand == True:
        for x in X:
            l = x.index(str(eos_id)) if str(eos_id) in x else len(x)
            Y.append(x[:l])
    else:
        Y = X

    f_out = open(out_file, "w")

    for y in Y:
        f_out.write(" ".join(y) + "\n")

    f_out.close()
    f.close()


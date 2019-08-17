import numpy as np
import cPickle as pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def get_pred_file(translations, data, name):

    id2word = pickle.load( open("data/%s/id2word.p" % (data, ), "rb") )
    
    f = open("output/" + name + ".txt", "w")

    for tran in translations:
        f.write(" ".join([id2word[i] for i in tran]) + "\n")


def calculate_metrics(cand, ref, ref_len, metric="bleu"):
    scores = []
    if metric == "bleu":
        for i in range(len(ref_len)):
            candidate = cand[i][:ref_len[i]].tolist()
            reference = [ref[i][:ref_len[i]].tolist()]
            chencherry = SmoothingFunction()
            score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
            scores.append(score)

    return np.asarray(scores, dtype=np.float32)

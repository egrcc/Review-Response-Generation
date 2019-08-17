import numpy as np
import collections
import cPickle as pickle
import os
import random
from gensim.models import word2vec


def bulid_vocab(dataset, vocab_size):

    f1 = open("dataset/%s/train_source.txt" % (dataset, ), "r")
    f2 = open("dataset/%s/valid_source.txt" % (dataset, ), "r")
    f1_2 = open("dataset/%s/train_source_2.txt" % (dataset, ), "r")
    f2_2 = open("dataset/%s/valid_source_2.txt" % (dataset, ), "r")
    f3 = open("dataset/%s/train_target.txt" % (dataset, ), "r")
    f4 = open("dataset/%s/valid_target.txt" % (dataset, ), "r")

    data = f1.read().split()
    data += f2.read().split()
    data += f1_2.read().split()
    data += f2_2.read().split()

    data_target = []
    for line in f3.readlines():
        line = ["<sos>"] + line.split() + ["<eos>"]
        data_target += line
    for line in f4.readlines():
        line = ["<sos>"] + line.split() + ["<eos>"]
        data_target += line

    data += data_target
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    words = list(words[:vocab_size-1])

    id_to_word = ["<unk>"] + words
    word_to_id = dict(zip(id_to_word, range(len(id_to_word))))

    pickle.dump( id_to_word, open("data/%s/id2word.p" % (dataset, ), "wb") )
    pickle.dump( word_to_id, open("data/%s/word2id.p" % (dataset, ), "wb") )


def bulid_field_vocab(dataset):

    f1 = open("dataset/%s/train_field.txt" % (dataset, ), "r")
    f2 = open("dataset/%s/valid_field.txt" % (dataset, ), "r")
    f3 = open("dataset/%s/test_field.txt" % (dataset, ), "r")

    data = f1.read().split()
    data += f2.read().split()
    data += f3.read().split()

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    words = list(words)

    id_to_word = words
    word_to_id = dict(zip(id_to_word, range(len(id_to_word))))

    pickle.dump( id_to_word, open("data/%s/id2field.p" % (dataset, ), "wb") )
    pickle.dump( word_to_id, open("data/%s/field2id.p" % (dataset, ), "wb") )


def bulid_pos_vocab(dataset):

    f1 = open("dataset/%s/train_pos1.txt" % (dataset, ), "r")
    f2 = open("dataset/%s/valid_pos1.txt" % (dataset, ), "r")
    f3 = open("dataset/%s/test_pos1.txt" % (dataset, ), "r")
    f4 = open("dataset/%s/train_pos2.txt" % (dataset, ), "r")
    f5 = open("dataset/%s/valid_pos2.txt" % (dataset, ), "r")
    f6 = open("dataset/%s/test_pos2.txt" % (dataset, ), "r")

    data = f1.read().split()
    data += f2.read().split()
    data += f3.read().split()
    data += f4.read().split()
    data += f5.read().split()
    data += f6.read().split()

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    words = list(words)

    id_to_word = words
    word_to_id = dict(zip(id_to_word, range(len(id_to_word))))

    pickle.dump( id_to_word, open("data/%s/id2pos.p" % (dataset, ), "wb") )
    pickle.dump( word_to_id, open("data/%s/pos2id.p" % (dataset, ), "wb") )


def get_embedding(dataset, vocab_size, embedding_size):
    if os.path.exists("data/%s/embedding_%d_%d.p" % (dataset, vocab_size, embedding_size)) == False:
        
        f1 = open("dataset/%s/train_source.txt" % (dataset, ), "r")
        f1_2 = open("dataset/%s/train_source_2.txt" % (dataset, ), "r")
        f2 = open("dataset/%s/train_target.txt" % (dataset, ), "r")
        f3 = open("dataset/%s/valid_source.txt" % (dataset, ), "r")
        f3_2 = open("dataset/%s/valid_source_2.txt" % (dataset, ), "r")
        f4 = open("dataset/%s/valid_target.txt" % (dataset, ), "r")
        
        data = []
        data.extend([line.split() for line in f1.readlines()])
        data.extend([line.split() for line in f1_2.readlines()])
        data.extend([["<sos>"] + line.split() + ["<eos>"] for line in f2.readlines()])
        data.extend([line.split() for line in f3.readlines()])
        data.extend([line.split() for line in f3_2.readlines()])
        data.extend([["<sos>"] + line.split() + ["<eos>"] for line in f4.readlines()])

        model = word2vec.Word2Vec(data, size=embedding_size, workers=16, min_count=1, window=8, iter=20)

        id2word = pickle.load( open("data/%s/id2word.p" % (dataset, ), "rb") )

        embedding = np.random.normal(0, 1, (len(id2word), embedding_size))

        for i in range(1, len(id2word)):
            embedding[i] = model[id2word[i]]

        pickle.dump( embedding, open("data/%s/embedding_%d_%d.p" % (dataset, vocab_size, embedding_size), "wb") )
    else:
        embedding = pickle.load( open("data/%s/embedding_%d_%d.p" % (dataset, vocab_size, embedding_size), "rb") )

    return embedding


def get_train_and_valid(dataset):

    if os.path.exists("data/%s/train_X.p" % (dataset, )) == False:

        word2id = pickle.load( open("data/%s/word2id.p" % (dataset, ), "rb") )
        unk_id = word2id["<unk>"]

        f_train_X = open("dataset/%s/train_source.txt" % (dataset, ), "r")
        f_train_Y = open("dataset/%s/train_target.txt" % (dataset, ), "r")
        f_valid_X = open("dataset/%s/valid_source.txt" % (dataset, ), "r")
        f_valid_Y = open("dataset/%s/valid_target.txt" % (dataset, ), "r")
        
        train_lines_X = f_train_X.readlines()
        train_lines_Y = []
        for line in f_train_Y.readlines():
            line = line.strip()
            line = "<sos> " + line + " <eos>"
            train_lines_Y.append(line)

        valid_lines_X = f_valid_X.readlines()
        valid_lines_Y = []
        for line in f_valid_Y.readlines():
            line = line.strip()
            line = "<sos> " + line + " <eos>"
            valid_lines_Y.append(line)

        train_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), train_lines_X)
        train_Y = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), train_lines_Y)

        valid_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), valid_lines_X)
        valid_Y = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), valid_lines_Y)

        pickle.dump( train_X, open("data/%s/train_X.p" % (dataset, ), "wb") )
        pickle.dump( train_Y, open("data/%s/train_Y.p" % (dataset, ), "wb") )
        pickle.dump( valid_X, open("data/%s/valid_X.p" % (dataset, ), "wb") )
        pickle.dump( valid_Y, open("data/%s/valid_Y.p" % (dataset, ), "wb") )
    else:
        train_X = pickle.load( open("data/%s/train_X.p" % (dataset, ), "rb") )
        train_Y = pickle.load( open("data/%s/train_Y.p" % (dataset, ), "rb") )
        valid_X = pickle.load( open("data/%s/valid_X.p" % (dataset, ), "rb") )
        valid_Y = pickle.load( open("data/%s/valid_Y.p" % (dataset, ), "rb") )

    return train_X, train_Y, valid_X, valid_Y


def get_train_and_valid_2(dataset):

    if os.path.exists("data/%s/train_X_2.p" % (dataset, )) == False:

        word2id = pickle.load( open("data/%s/word2id.p" % (dataset, ), "rb") )
        unk_id = word2id["<unk>"]

        f_train_X = open("dataset/%s/train_source_2.txt" % (dataset, ), "r")
        f_valid_X = open("dataset/%s/valid_source_2.txt" % (dataset, ), "r")
        
        train_lines_X = f_train_X.readlines()
        valid_lines_X = f_valid_X.readlines()
        

        train_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), train_lines_X)
        valid_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), valid_lines_X)

        pickle.dump( train_X, open("data/%s/train_X_2.p" % (dataset, ), "wb") )
        pickle.dump( valid_X, open("data/%s/valid_X_2.p" % (dataset, ), "wb") )
    else:
        train_X = pickle.load( open("data/%s/train_X_2.p" % (dataset, ), "rb") )
        valid_X = pickle.load( open("data/%s/valid_X_2.p" % (dataset, ), "rb") )

    return train_X, valid_X


def get_field_train_and_valid(dataset):

    if os.path.exists("data/%s/train_X_f.p" % (dataset, )) == False:

        field2id = pickle.load( open("data/%s/field2id.p" % (dataset, ), "rb") )

        f_train_X_f = open("dataset/%s/train_field.txt" % (dataset, ), "r")
        f_valid_X_f = open("dataset/%s/valid_field.txt" % (dataset, ), "r")
        
        train_lines_X_f = f_train_X_f.readlines()
        valid_lines_X_f = f_valid_X_f.readlines()

        train_X_f = map(lambda x: map(lambda y: field2id[y], x.split()), train_lines_X_f)
        valid_X_f = map(lambda x: map(lambda y: field2id[y], x.split()), valid_lines_X_f)

        pickle.dump( train_X_f, open("data/%s/train_X_f.p" % (dataset, ), "wb") )
        pickle.dump( valid_X_f, open("data/%s/valid_X_f.p" % (dataset, ), "wb") )
    else:
        train_X_f = pickle.load( open("data/%s/train_X_f.p" % (dataset, ), "rb") )
        valid_X_f = pickle.load( open("data/%s/valid_X_f.p" % (dataset, ), "rb") )

    return train_X_f, valid_X_f


def get_pos_train_and_valid(dataset):

    if os.path.exists("data/%s/train_X_pos1.p" % (dataset, )) == False:

        pos2id = pickle.load( open("data/%s/pos2id.p" % (dataset, ), "rb") )

        f_train_X_pos1 = open("dataset/%s/train_pos1.txt" % (dataset, ), "r")
        f_valid_X_pos1 = open("dataset/%s/valid_pos1.txt" % (dataset, ), "r")
        f_train_X_pos2 = open("dataset/%s/train_pos2.txt" % (dataset, ), "r")
        f_valid_X_pos2 = open("dataset/%s/valid_pos2.txt" % (dataset, ), "r")
        
        train_lines_X_pos1 = f_train_X_pos1.readlines()
        valid_lines_X_pos1 = f_valid_X_pos1.readlines()
        train_lines_X_pos2 = f_train_X_pos2.readlines()
        valid_lines_X_pos2 = f_valid_X_pos2.readlines()

        train_X_pos1 = map(lambda x: map(lambda y: pos2id[y], x.split()), train_lines_X_pos1)
        valid_X_pos1 = map(lambda x: map(lambda y: pos2id[y], x.split()), valid_lines_X_pos1)
        train_X_pos2 = map(lambda x: map(lambda y: pos2id[y], x.split()), train_lines_X_pos2)
        valid_X_pos2 = map(lambda x: map(lambda y: pos2id[y], x.split()), valid_lines_X_pos2)

        pickle.dump( train_X_pos1, open("data/%s/train_X_pos1.p" % (dataset, ), "wb") )
        pickle.dump( valid_X_pos1, open("data/%s/valid_X_pos1.p" % (dataset, ), "wb") )
        pickle.dump( train_X_pos2, open("data/%s/train_X_pos2.p" % (dataset, ), "wb") )
        pickle.dump( valid_X_pos2, open("data/%s/valid_X_pos2.p" % (dataset, ), "wb") )
    else:
        train_X_pos1 = pickle.load( open("data/%s/train_X_pos1.p" % (dataset, ), "rb") )
        valid_X_pos1 = pickle.load( open("data/%s/valid_X_pos1.p" % (dataset, ), "rb") )
        train_X_pos2 = pickle.load( open("data/%s/train_X_pos2.p" % (dataset, ), "rb") )
        valid_X_pos2 = pickle.load( open("data/%s/valid_X_pos2.p" % (dataset, ), "rb") )

    return train_X_pos1, valid_X_pos1, train_X_pos2, valid_X_pos2


def get_test(dataset):

    if os.path.exists("data/%s/test_X.p" % (dataset, )) == False:

        word2id = pickle.load( open("data/%s/word2id.p" % (dataset, ), "rb") )
        unk_id = word2id["<unk>"]

        f_test_X = open("dataset/%s/test_source.txt" % (dataset, ), "r")
        f_test_Y = open("dataset/%s/test_target.txt" % (dataset, ), "r")

        test_lines_X = f_test_X.readlines()
        test_lines_Y = []
        for line in f_test_Y.readlines():
            line = line.strip()
            line = "<sos> " + line + " <eos>"
            test_lines_Y.append(line)

        test_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), test_lines_X)
        test_Y = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), test_lines_Y)

        pickle.dump( test_X, open("data/%s/test_X.p" % (dataset, ), "wb") )
        pickle.dump( test_Y, open("data/%s/test_Y.p" % (dataset, ), "wb") )
    else:
        test_X = pickle.load( open("data/%s/test_X.p" % (dataset, ), "rb") )
        test_Y = pickle.load( open("data/%s/test_Y.p" % (dataset, ), "rb") )

    return test_X, test_Y


def get_test_2(dataset):

    if os.path.exists("data/%s/test_X_2.p" % (dataset, )) == False:

        word2id = pickle.load( open("data/%s/word2id.p" % (dataset, ), "rb") )
        unk_id = word2id["<unk>"]

        f_test_X = open("dataset/%s/test_source_2.txt" % (dataset, ), "r")

        test_lines_X = f_test_X.readlines()

        test_X = map(lambda x: map(lambda y: word2id.get(y, unk_id), x.split()), test_lines_X)

        pickle.dump( test_X, open("data/%s/test_X_2.p" % (dataset, ), "wb") )
    else:
        test_X = pickle.load( open("data/%s/test_X_2.p" % (dataset, ), "rb") )

    return test_X


def get_field_test(dataset):

    if os.path.exists("data/%s/test_X_f.p" % (dataset, )) == False:

        field2id = pickle.load( open("data/%s/field2id.p" % (dataset, ), "rb") )

        f_test_X_f = open("dataset/%s/test_field.txt" % (dataset, ), "r")

        test_lines_X_f = f_test_X_f.readlines()

        test_X_f = map(lambda x: map(lambda y: field2id[y], x.split()), test_lines_X_f)

        pickle.dump( test_X_f, open("data/%s/test_X_f.p" % (dataset, ), "wb") )
    else:
        test_X_f = pickle.load( open("data/%s/test_X_f.p" % (dataset, ), "rb") )

    return test_X_f


def get_pos_test(dataset):

    if os.path.exists("data/%s/test_X_pos1.p" % (dataset, )) == False:

        pos2id = pickle.load( open("data/%s/pos2id.p" % (dataset, ), "rb") )

        f_test_X_pos1 = open("dataset/%s/test_pos1.txt" % (dataset, ), "r")
        f_test_X_pos2 = open("dataset/%s/test_pos2.txt" % (dataset, ), "r")

        test_lines_X_pos1 = f_test_X_pos1.readlines()
        test_lines_X_pos2 = f_test_X_pos2.readlines()

        test_X_pos1 = map(lambda x: map(lambda y: pos2id[y], x.split()), test_lines_X_pos1)
        test_X_pos2 = map(lambda x: map(lambda y: pos2id[y], x.split()), test_lines_X_pos2)

        pickle.dump( test_X_pos1, open("data/%s/test_X_pos1.p" % (dataset, ), "wb") )
        pickle.dump( test_X_pos2, open("data/%s/test_X_pos2.p" % (dataset, ), "wb") )
    else:
        test_X_pos1 = pickle.load( open("data/%s/test_X_pos1.p" % (dataset, ), "rb") )
        test_X_pos2 = pickle.load( open("data/%s/test_X_pos2.p" % (dataset, ), "rb") )

    return test_X_pos1, test_X_pos2


def shuffle_list(a, b, c, d, e, f):
    """
    shuffle a, b, c, d, e, f simultaneously
    """
    z = list(zip(a, b, c, d, e, f))
    random.shuffle(z)
    a, b, c, d, e, f = zip(*z)

    return a, b, c, d, e, f


def padding(X):
    max_len = 0

    for x in X:
        if len(x) > max_len:
            max_len = len(x)

    padded_X = np.ones((len(X), max_len), dtype=np.int32) * 0

    len_ = 0
    X_len = []
    for i in range(len(X)):
        len_ += len(X[i])
        X_len.append(len(X[i]))
        for j in range(len(X[i])):
            padded_X[i, j] = X[i][j]

    return padded_X, X_len, len_


def data_iterator(X, X_2, X_f, X_pos1, X_pos2, Y, batch_size, shuffle=True):
    if shuffle == True:
        X, X_2, X_f, X_pos1, X_pos2, Y = shuffle_list(X, X_2, X_f, X_pos1, X_pos2, Y)

    Y_ipt = [y[:-1] for y in Y]
    Y_tgt = [y[1:] for y in Y]

    data_len = len(X)
    batch_len = data_len / batch_size

    for i in range(batch_len):
        batch_X = X[i*batch_size:(i+1)*batch_size]
        batch_X_2 = X_2[i*batch_size:(i+1)*batch_size]
        batch_X_f = X_f[i*batch_size:(i+1)*batch_size]
        batch_X_pos1 = X_pos1[i*batch_size:(i+1)*batch_size]
        batch_X_pos2 = X_pos2[i*batch_size:(i+1)*batch_size]
        batch_Y_ipt = Y_ipt[i*batch_size:(i+1)*batch_size]
        batch_Y_tgt = Y_tgt[i*batch_size:(i+1)*batch_size]

        padded_X, X_len, _ = padding(batch_X)
        padded_X_2, X_2_len, _ = padding(batch_X_2)
        padded_X_f, _, _ = padding(batch_X_f)
        padded_X_pos1, _, _ = padding(batch_X_pos1)
        padded_X_pos2, _, _ = padding(batch_X_pos2)
        padded_Y_ipt, Y_ipt_len, _ = padding(batch_Y_ipt)
        padded_Y_tgt, _, total_len_Y_tgt = padding(batch_Y_tgt)

        yield np.transpose(padded_X), np.array(X_len, dtype=np.int32), \
                np.transpose(padded_X_2), np.array(X_2_len, dtype=np.int32), \
                np.transpose(padded_X_f), np.transpose(padded_X_pos1), np.transpose(padded_X_pos2), \
                np.transpose(padded_Y_ipt), np.array(Y_ipt_len, dtype=np.int32), \
                np.transpose(padded_Y_tgt), total_len_Y_tgt

    if batch_len*batch_size != data_len:

        batch_X = X[batch_len*batch_size:]
        batch_X_2 = X_2[batch_len*batch_size:]
        batch_X_f = X_f[batch_len*batch_size:]
        batch_X_pos1 = X_pos1[batch_len*batch_size:]
        batch_X_pos2 = X_pos2[batch_len*batch_size:]
        batch_Y_ipt = Y_ipt[batch_len*batch_size:]
        batch_Y_tgt = Y_tgt[batch_len*batch_size:]

        padded_X, X_len, _ = padding(batch_X)
        padded_X_2, X_2_len, _ = padding(batch_X_2)
        padded_X_f, _, _ = padding(batch_X_f)
        padded_X_pos1, _, _ = padding(batch_X_pos1)
        padded_X_pos2, _, _ = padding(batch_X_pos2)
        padded_Y_ipt, Y_ipt_len, _ = padding(batch_Y_ipt)
        padded_Y_tgt, _, total_len_Y_tgt = padding(batch_Y_tgt)

        yield np.transpose(padded_X), np.array(X_len, dtype=np.int32), \
                np.transpose(padded_X_2), np.array(X_2_len, dtype=np.int32), \
                np.transpose(padded_X_f), np.transpose(padded_X_pos1), np.transpose(padded_X_pos2), \
                np.transpose(padded_Y_ipt), np.array(Y_ipt_len, dtype=np.int32), \
                np.transpose(padded_Y_tgt), total_len_Y_tgt


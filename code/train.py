import tensorflow as tf
import numpy as np
import utils_data
import cPickle as pickle
from utils import calculate_metrics
from model import PointerGeneratorModel, PointerGeneratorConfig
from model import CopynetModel, CopynetConfig
from model import Seq2seqKBFinalModel, Seq2seqKBFinalConfig


tf.flags.DEFINE_string("model", "copynet", "The model.")
tf.flags.DEFINE_string("decode", "training", "The decode.")
tf.flags.DEFINE_string("data", "review", "The dataset.")
tf.flags.DEFINE_string("name", "", "The model name.")
tf.flags.DEFINE_float("memory", 0.5, "Allowing GPU memory growth")
tf.flags.DEFINE_boolean("self_atte", False, "Set to True for using self attention encoder.")
FLAGS = tf.flags.FLAGS

if FLAGS.model == "copynet":
    config = CopynetConfig()
elif FLAGS.model == "pointer":
    config = PointerGeneratorConfig()
elif FLAGS.model == "seq2seqkbfinal":
    config = Seq2seqKBFinalConfig()

train_X, train_Y, valid_X, valid_Y = utils_data.get_train_and_valid(FLAGS.data)
test_X, test_Y = utils_data.get_test(FLAGS.data)

train_X_2, valid_X_2 = utils_data.get_train_and_valid_2(FLAGS.data)
test_X_2 = utils_data.get_test_2(FLAGS.data)

train_X_f, valid_X_f = utils_data.get_field_train_and_valid(FLAGS.data)
test_X_f = utils_data.get_field_test(FLAGS.data)

train_X_pos1, valid_X_pos1, train_X_pos2, valid_X_pos2 = utils_data.get_pos_train_and_valid(FLAGS.data)
test_X_pos1, test_X_pos2 = utils_data.get_pos_test(FLAGS.data)

word2id = pickle.load( open("data/%s/word2id.p" % FLAGS.data, "rb") )
vocab_size = len(word2id)
tgt_sos_id = word2id["<sos>"]
tgt_eos_id = word2id["<eos>"]

field2id = pickle.load( open("data/%s/field2id.p" % FLAGS.data, "rb") )
field_vocab_size = len(field2id)
pos2id = pickle.load( open("data/%s/pos2id.p" % FLAGS.data, "rb") )
pos_vocab_size = len(pos2id)

init_embedding = utils_data.get_embedding(FLAGS.data, vocab_size, config.embedding_size)

tfConfig = tf.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory

with tf.Graph().as_default(), tf.Session(config=tfConfig) as sess:

    with tf.variable_scope("Model", reuse=None):
        if FLAGS.model == "copynet":
            m = CopynetModel(vocab_size,
                            config.gen_vocab_size,
                            config.embedding_size,
                            config.source_hidden_size,
                            config.target_hidden_size,
                            config.attention_layer_size,
                            config.num_layers,
                            FLAGS.decode,
                            tgt_sos_id,
                            tgt_eos_id,
                            config.decode_iter,
                            config.max_grad_norm,
                            config.learning_rate,
                            init_embedding,
                            config.attention,
                            config.copy)
        elif FLAGS.model == "pointer":
            m = PointerGeneratorModel(vocab_size,
                            config.gen_vocab_size,
                            config.embedding_size,
                            config.source_hidden_size,
                            config.target_hidden_size,
                            config.attention_layer_size,
                            config.num_layers,
                            FLAGS.decode,
                            tgt_sos_id,
                            tgt_eos_id,
                            config.decode_iter,
                            config.max_grad_norm,
                            config.learning_rate,
                            init_embedding)
        elif FLAGS.model == "seq2seqkbfinal":
            m = Seq2seqKBFinalModel(vocab_size,
                            field_vocab_size,
                            pos_vocab_size,
                            config.gen_vocab_size,
                            config.embedding_size,
                            config.field_embedding_size,
                            config.pos_embedding_size,
                            config.source_hidden_size,
                            config.target_hidden_size,
                            config.attention_layer_size,
                            config.num_atte_layers,
                            config.num_heads,
                            config.num_layers,
                            FLAGS.decode,
                            tgt_sos_id,
                            tgt_eos_id,
                            config.decode_iter,
                            config.max_grad_norm,
                            config.learning_rate,
                            init_embedding,
                            rl=False,
                            seed=None,
                            self_atte=FLAGS.self_atte)
    with tf.variable_scope("Model", reuse=True):
        if FLAGS.model == "seq2seqkbfinal":
            m_train2 = Seq2seqKBFinalModel(vocab_size,
                            field_vocab_size,
                            pos_vocab_size,
                            config.gen_vocab_size,
                            config.embedding_size,
                            config.field_embedding_size,
                            config.pos_embedding_size,
                            config.source_hidden_size,
                            config.target_hidden_size,
                            config.attention_layer_size,
                            config.num_atte_layers,
                            config.num_heads,
                            config.num_layers,
                            FLAGS.decode,
                            tgt_sos_id,
                            tgt_eos_id,
                            config.decode_iter,
                            config.max_grad_norm,
                            config.learning_rate,
                            init_embedding,
                            rl=False,
                            seed=None,
                            self_atte=FLAGS.self_atte)
    with tf.variable_scope("Model", reuse=True):
        if FLAGS.model == "seq2seqkbfinal":
            m_valid = Seq2seqKBFinalModel(vocab_size,
                            field_vocab_size,
                            pos_vocab_size,
                            config.gen_vocab_size,
                            config.embedding_size,
                            config.field_embedding_size,
                            config.pos_embedding_size,
                            config.source_hidden_size,
                            config.target_hidden_size,
                            config.attention_layer_size,
                            config.num_atte_layers,
                            config.num_heads,
                            config.num_layers,
                            FLAGS.decode,
                            tgt_sos_id,
                            tgt_eos_id,
                            config.decode_iter,
                            config.max_grad_norm,
                            config.learning_rate,
                            init_embedding,
                            rl=False,
                            seed=None,
                            self_atte=FLAGS.self_atte)

    sess.run(tf.global_variables_initializer())

    best_valid_perplexity = 10000.
    pre_valid_perplexity = 10000.

    model_path = "model/%s_%s_%s_%s.ckpt" % (FLAGS.model, FLAGS.decode, FLAGS.data, FLAGS.name)
    saver = tf.train.Saver()

    cur_lr = config.learning_rate
    coverage_lambda = 0.
    step_num = 0

    for epoch in range(config.max_epochs):

        if FLAGS.model == "pointer" or FLAGS.model == "seq2seqkbfinal":
            if epoch == 15:
                coverage_lambda = 0.1

        total_cost = 0.
        total_step = 0
        total_len = 0

        valid_cost = 0.
        valid_step = 0
        valid_len = 0
        valid_score = 0.

        # train
        for step, (X, X_len, X_2, X_2_len, X_f, X_pos1, X_pos2,
                        Y, Y_len, Y_t, len_) in enumerate(utils_data.data_iterator(
                        train_X, train_X_2, train_X_f, train_X_pos1, train_X_pos2,
                        train_Y, config.batch_size, shuffle=True)):
            if FLAGS.model == "seq2seqkbfinal":
                cost, _ = sess.run([m.classify_loss, m.train_op],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.encoder_inputs_2: X_2,
                                    m.encoder_inputs_2_length: X_2_len,
                                    m.encoder_inputs_f: X_f,
                                    m.encoder_inputs_pos1: X_pos1,
                                    m.encoder_inputs_pos2: X_pos2,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.coverage_lambda: coverage_lambda,
                                    m.dropout_keep_prob: config.dropout_keep_prob})
            elif FLAGS.model == "pointer":
                cost, _ = sess.run([m.classify_loss, m.train_op],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.coverage_lambda: coverage_lambda,
                                    m.dropout_keep_prob: config.dropout_keep_prob})
            else:
                cost, _ = sess.run([m.classify_loss, m.train_op],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.dropout_keep_prob: config.dropout_keep_prob})
                

            cost = cost * len(X[0])
            
            total_cost += cost
            total_step += 1
            total_len += len_

            perplexity = np.exp(cost / len_)

            if step % 50 == 0:
                # print logits
                print("Epoch: %d Step: %d Cost: %.5f Perplexity: %.5f" % (epoch, step, cost, perplexity))
            if epoch >= 10 and step_num % 5000 == 0:
                cur_lr = cur_lr * config.lr_self_decay
                m.assign_lr(sess, cur_lr)
                print "------self decay------"
            step_num += 1

        avg_cost = total_cost / total_step
        total_perplexity = np.exp(total_cost / total_len)

        print("Epoch: %d Average cost: %.5f Perplexity: %.5f" % (epoch, avg_cost, total_perplexity))

        # valid
        for step, (X, X_len, X_2, X_2_len, X_f, X_pos1, X_pos2,
                Y, Y_len, Y_t, len_) in enumerate(utils_data.data_iterator(
                valid_X, valid_X_2, valid_X_f, valid_X_pos1, valid_X_pos2,
                valid_Y, config.batch_size, shuffle=False)):

            if FLAGS.model == "seq2seqkbfinal":
                cost, b_id = sess.run([m.classify_loss, m.baseline_sample_id],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.encoder_inputs_2: X_2,
                                    m.encoder_inputs_2_length: X_2_len,
                                    m.encoder_inputs_f: X_f,
                                    m.encoder_inputs_pos1: X_pos1,
                                    m.encoder_inputs_pos2: X_pos2,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.coverage_lambda: coverage_lambda,
                                    m.dropout_keep_prob: 1.})
            elif FLAGS.model == "pointer":
                cost, b_id = sess.run([m.classify_loss, m.baseline_sample_id],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.coverage_lambda: coverage_lambda,
                                    m.dropout_keep_prob: 1.})
            else:
                cost, b_id = sess.run([m.classify_loss, m.baseline_sample_id],
                        feed_dict={m.encoder_inputs: X,
                                    m.encoder_inputs_length: X_len,
                                    m.decoder_inputs: Y,
                                    m.decoder_inputs_length: Y_len,
                                    m.decoder_targets: Y_t,
                                    m.dropout_keep_prob: 1.})

            baseline = calculate_metrics(b_id, np.transpose(Y_t), Y_len, metric="bleu")
            valid_score += np.mean(baseline)
                
            cost = cost * len(X[0])

            valid_cost += cost
            valid_step += 1
            valid_len += len_

        avg_valid_score = valid_score / valid_step
        avg_cost = valid_cost / valid_step
        valid_perplexity = np.exp(valid_cost / valid_len)

        if valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            save_path = saver.save(sess, model_path)
            print("-----------------------------save model.------------------------------------")

        if valid_perplexity > pre_valid_perplexity:
            cur_lr = cur_lr * config.lr_decay
            m.assign_lr(sess, cur_lr)
            print("------------decay------------")

        pre_valid_perplexity = valid_perplexity

        print("Epoch: %d Validation Average cost: %.5f \
            Perplexity: %.5f Score: %.5f" % (epoch, avg_cost, valid_perplexity, avg_valid_score))

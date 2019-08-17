import tensorflow as tf
import numpy as np
import utils_data
import cPickle as pickle
from utils import calculate_metrics
from model import Seq2seqKBFinalModel, Seq2seqKBFinalConfig


tf.flags.DEFINE_string("model", "seq2seqkbfinal", "The model.")
tf.flags.DEFINE_string("decode", "training", "The decode.")
tf.flags.DEFINE_string("data", "review", "The dataset.")
tf.flags.DEFINE_string("name", "", "The model name.")
tf.flags.DEFINE_float("memory", 0.5, "Allowing GPU memory growth")
tf.flags.DEFINE_boolean("self_atte", False, "Set to True for using self attention encoder.")
FLAGS = tf.flags.FLAGS

if FLAGS.model == "seq2seqkbfinal":
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

    if FLAGS.model == "seq2seqkbfinal":
        with tf.variable_scope("Model", reuse=None):
            m_train1 = Seq2seqKBFinalModel(vocab_size,
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
                            rl=True,
                            seed=5,
                            self_atte=FLAGS.self_atte)
        with tf.variable_scope("Model", reuse=True):
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
                            rl=True,
                            seed=5,
                            self_atte=FLAGS.self_atte)
        with tf.variable_scope("Model", reuse=True):
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
                            rl=True,
                            seed=5,
                            self_atte=FLAGS.self_atte)

    sess.run(tf.global_variables_initializer())

    best_valid_score = 0.
    pre_valid_score = 0.

    model_path = "model/%s_%s_%s_%s.ckpt" % (FLAGS.model, FLAGS.decode, FLAGS.data, FLAGS.name)
    saver = tf.train.Saver()

    load_path = saver.restore(sess, "model/seq2seqkbfinal_training_review_.ckpt")

    cur_lr = config.learning_rate
    coverage_lambda = 0.
    step_num = 0
    rl_lambda = 0.

    for epoch in range(20):

        if epoch == 0:
            coverage_lambda = 0.1
            cur_lr = 0.0001
            m_train2.assign_lr(sess, cur_lr)
            print("----------RL training------------")

            rl_lambda = 0.99

        total_cost = 0.
        total_step = 0
        total_len = 0
        total_score = 0.

        valid_step = 0
        valid_score = 0.
        valid_cost = 0.
        valid_len = 0

        # train
        for step, (X, X_len, X_2, X_2_len, X_f, X_pos1, X_pos2,
                        Y, Y_len, Y_t, len_) in enumerate(utils_data.data_iterator(
                        train_X, train_X_2, train_X_f, train_X_pos1, train_X_pos2,
                        train_Y, config.batch_size, shuffle=True)):
        
            if FLAGS.model == "seq2seqkbfinal":
                b_id, r_id = sess.run([m_train1.baseline_sample_id, m_train1.train_sample_id],
                            feed_dict={m_train1.encoder_inputs: X,
                                        m_train1.encoder_inputs_length: X_len,
                                        m_train1.encoder_inputs_2: X_2,
                                        m_train1.encoder_inputs_2_length: X_2_len,
                                        m_train1.encoder_inputs_f: X_f,
                                        m_train1.encoder_inputs_pos1: X_pos1,
                                        m_train1.encoder_inputs_pos2: X_pos2,
                                        m_train1.decoder_inputs: Y,
                                        m_train1.dropout_keep_prob: config.dropout_keep_prob})
     

            baseline = calculate_metrics(b_id, np.transpose(Y_t), Y_len, metric="bleu")
            rewards = calculate_metrics(r_id, np.transpose(Y_t), Y_len, metric="bleu")

            if FLAGS.model == "seq2seqkbfinal":
                cost, _ = sess.run([m_train2.classify_loss, m_train2.train_op],
                            feed_dict={m_train2.encoder_inputs: X,
                                        m_train2.encoder_inputs_length: X_len,
                                        m_train2.encoder_inputs_2: X_2,
                                        m_train2.encoder_inputs_2_length: X_2_len,
                                        m_train2.encoder_inputs_f: X_f,
                                        m_train2.encoder_inputs_pos1: X_pos1,
                                        m_train2.encoder_inputs_pos2: X_pos2,
                                        m_train2.decoder_inputs: Y,
                                        m_train2.decoder_inputs_length: Y_len,
                                        m_train2.decoder_targets: Y_t,
                                        m_train2.rewards: rewards,
                                        m_train2.baseline: baseline,
                                        m_train2.rl_lambda: rl_lambda,
                                        m_train2.coverage_lambda: coverage_lambda,
                                        m_train2.dropout_keep_prob: config.dropout_keep_prob})
          
            cost = cost * len(X[0])

            total_cost += cost
            total_len += len_
            total_score += np.mean(baseline)
            total_step += 1
           
            perplexity = np.exp(cost / len_)

            if step % 50 == 0:
                print("Epoch: %d Step: %d Cost: %.5f Perplexity: %.5f Metric Score: %.5f" \
                         % (epoch, step, cost, perplexity, np.mean(baseline)))

            # if epoch > 5 and step_num % 800 == 0 and cur_lr > 0.0005:
            #     cur_lr = cur_lr * config.lr_self_decay
            #     m_train2.assign_lr(sess, cur_lr)
            #     print "------self decay------"
            step_num += 1
            
        avg_cost = total_cost / total_step
        avg_score = total_score / total_step
        total_perplexity = np.exp(total_cost / total_len)

        print("Epoch: %d Average cost: %.5f Perplexity: %.5f Average Metric Score: %.5f" \
                    % (epoch, avg_cost, total_perplexity, avg_score))

        # valid
        for step, (X, X_len, X_2, X_2_len, X_f, X_pos1, X_pos2,
                Y, Y_len, Y_t, len_) in enumerate(utils_data.data_iterator(
                valid_X, valid_X_2, valid_X_f, valid_X_pos1, valid_X_pos2,
                valid_Y, config.batch_size, shuffle=False)):
        
            if FLAGS.model == "seq2seqkbfinal":
                b_id, cost = sess.run([m_valid.baseline_sample_id, m_valid.classify_loss],
                            feed_dict={m_valid.encoder_inputs: X,
                                        m_valid.encoder_inputs_length: X_len,
                                        m_valid.encoder_inputs_2: X_2,
                                        m_valid.encoder_inputs_2_length: X_2_len,
                                        m_valid.encoder_inputs_f: X_f,
                                        m_valid.encoder_inputs_pos1: X_pos1,
                                        m_valid.encoder_inputs_pos2: X_pos2,
                                        m_valid.decoder_inputs: Y,
                                        m_valid.decoder_inputs_length: Y_len,
                                        m_valid.decoder_targets: Y_t,
                                        m_valid.dropout_keep_prob: 1.})
        
            baseline = calculate_metrics(b_id, np.transpose(Y_t), Y_len, metric="bleu")

            valid_score += np.sum(baseline)
            valid_step += len(baseline)
            valid_cost += cost * len(X[0])
            valid_len += len_

        avg_valid_score = valid_score / valid_step
        valid_perplexity = np.exp(valid_cost / valid_len)

        if avg_valid_score > best_valid_score:
            best_valid_score = avg_valid_score
            save_path = saver.save(sess, model_path)
            print("-----------------------------save model.------------------------------------")

        if avg_valid_score < pre_valid_score:
            cur_lr = cur_lr * config.lr_decay
            m_train2.assign_lr(sess, cur_lr)
            print("---------------decay----------------")

        pre_valid_score = avg_valid_score

        print("Epoch: %d Perplexity: %.5f Validation Average Score: %.5f" \
                % (epoch, valid_perplexity, avg_valid_score))

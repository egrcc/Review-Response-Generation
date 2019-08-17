import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import utils
import utils_data
import cPickle as pickle
from model import PointerGeneratorModel, PointerGeneratorConfig
from model import CopynetModel, CopynetConfig
from model import Seq2seqKBFinalModel, Seq2seqKBFinalConfig


tf.flags.DEFINE_string("model", "copynet", "The model.")
tf.flags.DEFINE_string("decode", "greedy", "The decode.")
tf.flags.DEFINE_string("data", "review", "The dataset.")
tf.flags.DEFINE_string("name", "", "The model name.")
tf.flags.DEFINE_float("memory", 0.5, "Allowing GPU memory growth")
tf.flags.DEFINE_boolean("rl", False, "Set to True for using rl.")
tf.flags.DEFINE_boolean("self_atte", False, "Set to True for using self attention encoder.")
FLAGS = tf.flags.FLAGS

if FLAGS.model == "copynet":
    config = CopynetConfig()
elif FLAGS.model == "pointer":
    config = PointerGeneratorConfig()
elif FLAGS.model == "seq2seqkbfinal":
    config = Seq2seqKBFinalConfig()

test_X, test_Y = utils_data.get_test(FLAGS.data)
test_X_2 = utils_data.get_test_2(FLAGS.data)
test_X_f = utils_data.get_field_test(FLAGS.data)
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
                            rl=FLAGS.rl,
                            seed=None,
                            self_atte=FLAGS.self_atte)

    model_path = "model/%s_%s_%s_%s.ckpt" % (FLAGS.model, "training", FLAGS.data, FLAGS.name)
    saver = tf.train.Saver()

    load_path = saver.restore(sess, model_path)

    all_trans = []

    for step, (X, X_len, X_2, X_2_len, X_f, X_pos1, X_pos2,
                Y, Y_len, Y_t, len_) in enumerate(utils_data.data_iterator(
                test_X, test_X_2, test_X_f, test_X_pos1, test_X_pos2,
                test_Y, 256, shuffle=False)):

        if FLAGS.model == "seq2seqkbfinal":
        	translations = sess.run(m.translations, feed_dict={m.encoder_inputs: X,
                                                                m.encoder_inputs_length: X_len,
                                                                m.encoder_inputs_2: X_2,
                                                                m.encoder_inputs_2_length: X_2_len,
                                                                m.encoder_inputs_f: X_f,
							                                    m.encoder_inputs_pos1: X_pos1,
							                                    m.encoder_inputs_pos2: X_pos2,
                                                                m.dropout_keep_prob: 1.})
        else:
            translations = sess.run(m.translations, feed_dict={m.encoder_inputs: X,
                                                                m.encoder_inputs_length: X_len,
                                                                m.dropout_keep_prob: 1.})
            
        # translations = translations[:, :, 0]
        translations = translations[:, :, -1]

        all_trans.extend(translations.tolist())

    name = "%s_%s_%s_%s" % (FLAGS.model, FLAGS.decode, FLAGS.data, FLAGS.name)
    utils.get_pred_file(all_trans, FLAGS.data, name)

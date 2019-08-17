import tensorflow as tf
from modules import *
from copynet import CopyNetWrapper
from pointer_generator import PointerGeneratorWrapper
from seq2seq_kb_final import Seq2seqKBFinalWrapper


class CopynetConfig(object):

    batch_size = 16
    max_epochs = 30
    gen_vocab_size = 15000
    dropout_keep_prob = 0.85
    lr_decay = 0.8
    lr_self_decay = 0.8
    embedding_size = 128
    source_hidden_size = 256
    target_hidden_size = 256
    attention_layer_size = 256
    num_layers = 2
    max_grad_norm = 5
    learning_rate = 0.02
    decode_iter = 200
    attention = True
    copy = True


class CopynetModel(object):

    def __init__(self,
                vocab_size,
                gen_vocab_size,
                embedding_size,
                source_hidden_size,
                target_hidden_size,
                attention_layer_size,
                num_layers,
                decode,
                tgt_sos_id,
                tgt_eos_id,
                decode_iter,
                max_grad_norm,
                learning_rate,
                init_embedding,
                attention,
                copy):

        self.encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        self.batch_size = tf.shape(self.encoder_inputs)[1]
        self.decoder_max_length = tf.shape(self.decoder_inputs)[0]
        
        self.target_weights = tf.sequence_mask(self.decoder_inputs_length, self.decoder_max_length, dtype=tf.float32)

        with tf.variable_scope('encoder_embedding'):
            self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp, self.dropout_keep_prob)

        with tf.variable_scope('encoder_lstm'):
            
            def encoder_lstm_cell():
                return tf.contrib.rnn.GRUCell(source_hidden_size)

            def encoder_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(encoder_lstm_cell(), output_keep_prob=self.dropout_keep_prob)

            self.encoder_fw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])
            self.encoder_bw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])

            (encoder_outputs, encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                                                        cell_fw=self.encoder_fw_multi_cell,
                                                        cell_bw=self.encoder_bw_multi_cell,
                                                        inputs=encoder_emb_inp,
                                                        sequence_length=self.encoder_inputs_length,
                                                        time_major=True,
                                                        dtype=tf.float32
                                                    )

            self.encoder_outputs = tf.concat(encoder_outputs, 2)

            with tf.variable_scope('transition'):
                encoder_final_state = tf.concat(encoder_final_state, 2)
                transition_w = tf.get_variable("transition_w", [source_hidden_size * 2, target_hidden_size])
                transition_b = tf.get_variable("transition_b", [target_hidden_size])
                encoder_final_state = tf.einsum("aij,jk->aik", encoder_final_state, transition_w) + transition_b
                self.encoder_final_state = []
                for i in range(num_layers):
                    self.encoder_final_state.append(encoder_final_state[i])
                self.encoder_final_state = tuple(self.encoder_final_state)

        with tf.variable_scope('decoder_embedding'):
            decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
            decoder_emb_inp = tf.nn.dropout(decoder_emb_inp, self.dropout_keep_prob)

        with tf.variable_scope('decoder_lstm'):
            
            def decoder_lstm_cell():
                return tf.contrib.rnn.GRUCell(target_hidden_size)

            def decoder_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(decoder_lstm_cell(), output_keep_prob=self.dropout_keep_prob)

            decoder_multi_cell = tf.contrib.rnn.MultiRNNCell([decoder_attn_cell() for _ in range(num_layers)])

            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            decoder_initial_state = self.encoder_final_state
            memory_sequence_length = self.encoder_inputs_length
            encoder_inputs_id = tf.transpose(self.encoder_inputs, [1, 0])

            if attention == True:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    attention_layer_size, attention_states,
                    memory_sequence_length=memory_sequence_length, normalize=True)

                atte_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_multi_cell, attention_mechanism,
                    attention_layer_size=attention_layer_size)
                decoder_cell = CopyNetWrapper(atte_decoder_cell, attention_states,
                    encoder_inputs_id, memory_sequence_length, vocab_size, copy, gen_vocab_size)
            else:
                decoder_cell = CopyNetWrapper(decoder_multi_cell, attention_states,
                    encoder_inputs_id, memory_sequence_length, vocab_size, copy, gen_vocab_size)

        with tf.variable_scope('decoder'):

            
            if attention == True:
                wrapper_state = atte_decoder_cell.zero_state(batch_size=self.batch_size,
                                    dtype=tf.float32).clone(cell_state=decoder_initial_state)
                wrapper_state = decoder_cell.zero_state(batch_size=self.batch_size,
                                    dtype=tf.float32).clone(cell_state=wrapper_state)
            else:
                wrapper_state = decoder_cell.zero_state(batch_size=self.batch_size,
                                    dtype=tf.float32).clone(cell_state=decoder_initial_state)


            if decode == "training":

                baseline_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                baseline_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, baseline_helper, wrapper_state)

                baseline_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        baseline_decoder, maximum_iterations=self.decoder_max_length)

                self.baseline_sample_id = baseline_decoder_outputs.sample_id


                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.decoder_inputs_length, time_major=True)

                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, train_helper, wrapper_state)

                train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)

                with tf.variable_scope('loss'):

                    self.logits = train_decoder_outputs.rnn_output
                    
                    self.classify_loss = tf.contrib.seq2seq.sequence_loss(
                        self.logits,
                        tf.transpose(self.decoder_targets),
                        self.target_weights,
                        average_across_timesteps=False,
                        average_across_batch=False
                    )

                    self.classify_loss = tf.reduce_sum(self.classify_loss) / tf.cast(self.batch_size, tf.float32)

                    self.loss = self.classify_loss

                with tf.variable_scope('train_op'):
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

                    self.lr = tf.Variable(learning_rate, trainable=False)
                    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                    self.lr_update = tf.assign(self.lr, self.new_lr)

                    self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
                    self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

            elif decode == "greedy":

                generation_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                generation_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, generation_helper, wrapper_state)

                generation_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                    generation_decoder, maximum_iterations=decode_iter)

                self.translations = tf.expand_dims(generation_decoder_outputs.sample_id, -1)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


class PointerGeneratorConfig(object):

    batch_size = 16
    max_epochs = 30
    gen_vocab_size = 15000
    dropout_keep_prob = 0.85
    lr_decay = 0.8
    lr_self_decay = 0.8
    embedding_size = 128
    source_hidden_size = 256
    target_hidden_size = 256
    attention_layer_size = 256
    num_layers = 2
    max_grad_norm = 5
    learning_rate = 0.02
    decode_iter = 200


class PointerGeneratorModel(object):

    def __init__(self,
                vocab_size,
                gen_vocab_size,
                embedding_size,
                source_hidden_size,
                target_hidden_size,
                attention_layer_size,
                num_layers,
                decode,
                tgt_sos_id,
                tgt_eos_id,
                decode_iter,
                max_grad_norm,
                learning_rate,
                init_embedding):

        self.encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        self.coverage_lambda = tf.placeholder(dtype=tf.float32, name='coverage_lambda')

        self.batch_size = tf.shape(self.encoder_inputs)[1]
        self.decoder_max_length = tf.shape(self.decoder_inputs)[0]
        
        self.target_weights = tf.sequence_mask(self.decoder_inputs_length, self.decoder_max_length, dtype=tf.float32)

        with tf.variable_scope('encoder_embedding'):
            self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding', trainable=True)
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp, self.dropout_keep_prob)

        with tf.variable_scope('encoder_lstm'):
            
            def encoder_lstm_cell():
                return tf.contrib.rnn.GRUCell(source_hidden_size)

            def encoder_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(encoder_lstm_cell(), output_keep_prob=self.dropout_keep_prob)

            self.encoder_fw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])
            self.encoder_bw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])

            (encoder_outputs, encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                                                        cell_fw=self.encoder_fw_multi_cell,
                                                        cell_bw=self.encoder_bw_multi_cell,
                                                        inputs=encoder_emb_inp,
                                                        sequence_length=self.encoder_inputs_length,
                                                        time_major=True,
                                                        dtype=tf.float32
                                                    )

            self.encoder_outputs = tf.concat(encoder_outputs, 2)

            with tf.variable_scope('transition'):
                encoder_final_state = tf.concat(encoder_final_state, 2)
                transition_w = tf.get_variable("transition_w", [source_hidden_size * 2, target_hidden_size])
                transition_b = tf.get_variable("transition_b", [target_hidden_size])
                encoder_final_state = tf.einsum("aij,jk->aik", encoder_final_state, transition_w) + transition_b
                self.encoder_final_state = []
                for i in range(num_layers):
                    self.encoder_final_state.append(encoder_final_state[i])
                self.encoder_final_state = tuple(self.encoder_final_state)

        with tf.variable_scope('decoder_embedding'):
            decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
            decoder_emb_inp = tf.nn.dropout(decoder_emb_inp, self.dropout_keep_prob)

        with tf.variable_scope('decoder_lstm'):
            
            def decoder_lstm_cell():
                return tf.contrib.rnn.GRUCell(target_hidden_size)

            def decoder_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(decoder_lstm_cell(), output_keep_prob=self.dropout_keep_prob)

            decoder_multi_cell = tf.contrib.rnn.MultiRNNCell([decoder_attn_cell() for _ in range(num_layers)])

            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            decoder_initial_state = self.encoder_final_state
            memory_sequence_length = self.encoder_inputs_length
            memory_inputs_id = tf.transpose(self.encoder_inputs, [1, 0])

            decoder_cell = PointerGeneratorWrapper(decoder_multi_cell, attention_states, memory_inputs_id,
                                            memory_sequence_length, vocab_size,
                                            attention_layer_size, target_hidden_size, gen_vocab_size)

        with tf.variable_scope('decoder'):

            wrapper_state = decoder_cell.zero_state(batch_size=self.batch_size,
                                dtype=tf.float32).clone(cell_state=decoder_initial_state)

            if decode == "training":

                baseline_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                baseline_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, baseline_helper, wrapper_state)

                baseline_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        baseline_decoder, maximum_iterations=self.decoder_max_length)

                self.baseline_sample_id = baseline_decoder_outputs.sample_id

                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.decoder_inputs_length, time_major=True)

                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, train_helper, wrapper_state)

                train_decoder_outputs, train_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)

                with tf.variable_scope('loss'):

                    self.covloss = self.target_weights * (tf.transpose(tf.squeeze(train_decoder_state.covloss.stack())))
                    self.covloss = tf.reduce_sum(self.covloss) / tf.cast(self.batch_size, tf.float32)

                    self.classify_loss = tf.contrib.seq2seq.sequence_loss(
                        train_decoder_outputs.rnn_output,
                        tf.transpose(self.decoder_targets),
                        self.target_weights,
                        average_across_timesteps=False,
                        average_across_batch=False
                    )

                    self.classify_loss = tf.reduce_sum(self.classify_loss) / tf.cast(self.batch_size, tf.float32)

                    self.loss = self.classify_loss + self.coverage_lambda * self.covloss

                with tf.variable_scope('train_op'):
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

                    self.lr = tf.Variable(learning_rate, trainable=False)
                    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                    self.lr_update = tf.assign(self.lr, self.new_lr)

                    self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
                    self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

            elif decode == "greedy":

                generation_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                generation_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, generation_helper, wrapper_state)

                generation_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                    generation_decoder, maximum_iterations=decode_iter)

                self.translations = tf.expand_dims(generation_decoder_outputs.sample_id, -1)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


class Seq2seqKBFinalConfig(object):

    batch_size = 16
    max_epochs = 30
    gen_vocab_size = 15000
    dropout_keep_prob = 0.85
    lr_decay = 0.8
    lr_self_decay = 0.8
    embedding_size = 128
    field_embedding_size = 32
    pos_embedding_size = 8
    source_hidden_size = 256
    target_hidden_size = 256
    attention_layer_size = 256
    num_layers = 2
    max_grad_norm = 5
    learning_rate = 0.02
    decode_iter = 200
    # self-atte
    num_atte_layers = 6
    num_heads = 8


class Seq2seqKBFinalModel(object):

    def __init__(self,
                vocab_size,
                field_vocab_size,
                pos_vocab_size,
                gen_vocab_size,
                embedding_size,
                field_embedding_size,
                pos_embedding_size,
                source_hidden_size,
                target_hidden_size,
                attention_layer_size,
                num_atte_layers,
                num_heads,
                num_layers,
                decode,
                tgt_sos_id,
                tgt_eos_id,
                decode_iter,
                max_grad_norm,
                learning_rate,
                init_embedding,
                rl=False,
                seed=None,
                self_atte=False):

        self.encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_length')
        self.encoder_inputs_2 = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs_2')
        self.encoder_inputs_2_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_2_length')
        self.encoder_inputs_f = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs_f')
        self.encoder_inputs_pos1 = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs_pos1')
        self.encoder_inputs_pos2 = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs_pos2')
        self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        self.rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        self.baseline = tf.placeholder(shape=[None], dtype=tf.float32, name="baseline")
        self.rl_lambda = tf.placeholder(dtype=tf.float32, name="rl_lambda")
        self.coverage_lambda = tf.placeholder(dtype=tf.float32, name='coverage_lambda')

        if self_atte == True:
            self.encoder_inputs_t = tf.transpose(self.encoder_inputs)
            self.encoder_inputs_2_t = tf.transpose(self.encoder_inputs_2)
            self.encoder_inputs_f_t = tf.transpose(self.encoder_inputs_f)
            self.encoder_inputs_pos1_t = tf.transpose(self.encoder_inputs_pos1)
            self.encoder_inputs_pos2_t = tf.transpose(self.encoder_inputs_pos2)
            self.decoder_inputs_t = tf.transpose(self.decoder_inputs)
            self.decoder_targets_t = tf.transpose(self.decoder_targets)

        self.batch_size = tf.shape(self.encoder_inputs)[1]
        self.encoder_max_length = tf.shape(self.encoder_inputs)[0]
        self.encoder_2_max_length = tf.shape(self.encoder_inputs_2)[0]
        self.decoder_max_length = tf.shape(self.decoder_inputs)[0]
        self.source_weights = tf.sequence_mask(self.encoder_inputs_length, self.encoder_max_length, dtype=tf.float32)
        self.source_2_weights = tf.sequence_mask(self.encoder_inputs_2_length, self.encoder_2_max_length, dtype=tf.float32)
        self.target_weights = tf.sequence_mask(self.decoder_inputs_length, self.decoder_max_length, dtype=tf.float32)


        with tf.variable_scope('encoder_embedding'):
            self.embedding = tf.get_variable('embedding',
                                        dtype=tf.float32,
                                        initializer=tf.constant(init_embedding, dtype=tf.float32),
                                        trainable=True)
            self.field_embedding = tf.get_variable('field_embedding',
                                                    [field_vocab_size, field_embedding_size],
                                                    dtype=tf.float32)
            self.pos_embedding = tf.get_variable('pos_embedding',
                                                [pos_vocab_size, pos_embedding_size],
                                                dtype=tf.float32)
            if self_atte == True:
                encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs_t)
                encoder_emb_inp += positional_encoding(self.encoder_inputs_t,
                                                        num_units=embedding_size,
                                                        scale=False,
                                                        scope="enc_pe")
            else:
                encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp, self.dropout_keep_prob, seed=seed)
            
            if self_atte == True:
                encoder_emb_inp_2 = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs_2_t)
                encoder_emb_inp_2 += positional_encoding(self.encoder_inputs_2_t,
                                                        num_units=embedding_size,
                                                        scale=False,
                                                        scope="enc_2_pe")
                encoder_emb_inp_f = tf.nn.embedding_lookup(self.field_embedding, self.encoder_inputs_f_t)
                encoder_emb_inp_pos1 = tf.nn.embedding_lookup(self.pos_embedding, self.encoder_inputs_pos1_t)
                encoder_emb_inp_pos2 = tf.nn.embedding_lookup(self.pos_embedding, self.encoder_inputs_pos2_t)

            else:
                encoder_emb_inp_2 = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs_2)
                encoder_emb_inp_f = tf.nn.embedding_lookup(self.field_embedding, self.encoder_inputs_f)
                encoder_emb_inp_pos1 = tf.nn.embedding_lookup(self.pos_embedding, self.encoder_inputs_pos1)
                encoder_emb_inp_pos2 = tf.nn.embedding_lookup(self.pos_embedding, self.encoder_inputs_pos2)
            encoder_emb_inp_2 = tf.concat([encoder_emb_inp_2,
                                            encoder_emb_inp_f,
                                            encoder_emb_inp_pos1,
                                            encoder_emb_inp_pos2], axis=-1)
            encoder_emb_inp_2 = tf.nn.dropout(encoder_emb_inp_2, self.dropout_keep_prob, seed=seed)

        if self_atte == True:
            emb_W = tf.layers.Dense(2*source_hidden_size, use_bias=True, activation=tf.tanh, name='emb_W')
            emb_W_2 = tf.layers.Dense(2*source_hidden_size, use_bias=True, activation=tf.tanh, name='emb_W_2')
            self.enc = emb_W(encoder_emb_inp)
            self.enc_2 = emb_W_2(encoder_emb_inp_2)

        if self_atte == True:
            with tf.variable_scope('encoder_blocks'):
                for i in range(num_atte_layers):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                        keys=self.enc,
                                                        query_masks=self.source_weights,
                                                        key_masks=self.source_weights,
                                                        num_units=2*source_hidden_size,
                                                        num_heads=num_heads,
                                                        dropout_keep_prob=self.dropout_keep_prob,
                                                        rl=rl,
                                                        seed=seed,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[8*source_hidden_size, 2*source_hidden_size])
            with tf.variable_scope('encoder_2_blocks'):
                for i in range(num_atte_layers):
                    with tf.variable_scope("num_blocks_2_{}".format(i)):
                        ### Multihead Attention
                        self.enc_2 = multihead_attention(queries=self.enc_2,
                                                        keys=self.enc_2,
                                                        query_masks=self.source_2_weights,
                                                        key_masks=self.source_2_weights,
                                                        num_units=2*source_hidden_size,
                                                        num_heads=num_heads,
                                                        dropout_keep_prob=self.dropout_keep_prob,
                                                        rl=rl,
                                                        seed=seed,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc_2 = feedforward(self.enc_2, num_units=[8*source_hidden_size, 2*source_hidden_size])

            self.encoder_outputs = tf.transpose(self.enc, [1, 0, 2])
            self.encoder_outputs_2 = tf.transpose(self.enc_2, [1, 0, 2])

            with tf.variable_scope('transition'):
                encoder_final_state = tf.einsum("ijk,ij->ik", self.enc, self.source_weights) \
                                         / tf.reduce_sum(self.source_weights, axis=1, keepdims=True)
                encoder_final_state_2 = tf.einsum("ijk,ij->ik", self.enc_2, self.source_2_weights) \
                                        / tf.reduce_sum(self.source_2_weights, axis=1, keepdims=True)
                encoder_final_state_toal = tf.concat([encoder_final_state, encoder_final_state_2], 1)

                self.encoder_final_state = []
                for i in range(num_layers):
                    transition_w = tf.get_variable("transition_w_{}".format(i),
                                [source_hidden_size * 4, target_hidden_size])
                    transition_b = tf.get_variable("transition_b_{}".format(i), 
                                [target_hidden_size])
                    self.encoder_final_state.append(tf.einsum("aj,jk->ak", \
                        encoder_final_state_toal, transition_w) + transition_b)
                self.encoder_final_state = tuple(self.encoder_final_state)

        else:
            with tf.variable_scope('encoder_lstm'):
                
                def encoder_lstm_cell():
                    return tf.contrib.rnn.GRUCell(source_hidden_size)

                def encoder_attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(encoder_lstm_cell(),
                        output_keep_prob=self.dropout_keep_prob, seed=seed)
        
                with tf.variable_scope('review_lstm'):

                    self.encoder_fw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])
                    self.encoder_bw_multi_cell = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])
                
                    (encoder_outputs, encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                                                                cell_fw=self.encoder_fw_multi_cell,
                                                                cell_bw=self.encoder_bw_multi_cell,
                                                                inputs=encoder_emb_inp,
                                                                sequence_length=self.encoder_inputs_length,
                                                                time_major=True,
                                                                dtype=tf.float32
                                                            )
                with tf.variable_scope('aspect_lstm'):

                    self.encoder_fw_multi_cell_2 = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])
                    self.encoder_bw_multi_cell_2 = tf.contrib.rnn.MultiRNNCell([encoder_attn_cell() for _ in range(num_layers)])

                    (encoder_outputs_2, encoder_final_state_2) = tf.nn.bidirectional_dynamic_rnn(
                                                                cell_fw=self.encoder_fw_multi_cell_2,
                                                                cell_bw=self.encoder_bw_multi_cell_2,
                                                                inputs=encoder_emb_inp_2,
                                                                sequence_length=self.encoder_inputs_2_length,
                                                                time_major=True,
                                                                dtype=tf.float32
                                                            )

                self.encoder_outputs = tf.concat(encoder_outputs, 2)
                self.encoder_outputs_2 = tf.concat(encoder_outputs_2, 2)

                with tf.variable_scope('transition'):
                    encoder_final_state = tf.concat(encoder_final_state, 2)
                    encoder_final_state_2 = tf.concat(encoder_final_state_2, 2)

                    encoder_final_state = tf.concat([encoder_final_state, encoder_final_state_2], 2)

                    transition_w = tf.get_variable("transition_w", [source_hidden_size * 4, target_hidden_size])
                    transition_b = tf.get_variable("transition_b", [target_hidden_size])
                    encoder_final_state = tf.einsum("aij,jk->aik", encoder_final_state, transition_w) + transition_b
                    self.encoder_final_state = []
                    for i in range(num_layers):
                        self.encoder_final_state.append(encoder_final_state[i])
                    self.encoder_final_state = tuple(self.encoder_final_state)

        with tf.variable_scope('decoder_embedding'):
            decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)
            decoder_emb_inp = tf.nn.dropout(decoder_emb_inp, self.dropout_keep_prob, seed=seed)

        with tf.variable_scope('decoder_lstm'):
            
            def decoder_lstm_cell():
                return tf.contrib.rnn.GRUCell(target_hidden_size)

            def decoder_attn_cell():
                return tf.contrib.rnn.DropoutWrapper(decoder_lstm_cell(),
                    output_keep_prob=self.dropout_keep_prob, seed=seed)

            decoder_multi_cell = tf.contrib.rnn.MultiRNNCell([decoder_attn_cell() for _ in range(num_layers)])

            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            decoder_initial_state = self.encoder_final_state
            memory_sequence_length = self.encoder_inputs_length
            memory_inputs_id = tf.transpose(self.encoder_inputs, [1, 0])
            memory_inputs_2_id = tf.transpose(self.encoder_inputs_2, [1, 0])

            attention_states_2 = tf.transpose(self.encoder_outputs_2, [1, 0, 2])
            memory_sequence_2_length = self.encoder_inputs_2_length

            decoder_cell = Seq2seqKBFinalWrapper(decoder_multi_cell, attention_states, attention_states_2,
                                            memory_inputs_id, memory_inputs_2_id, memory_sequence_length, 
                                            memory_sequence_2_length, vocab_size, attention_layer_size, 
                                            source_hidden_size, target_hidden_size, gen_vocab_size)

        with tf.variable_scope('decoder'):
            
            wrapper_state = decoder_cell.zero_state(batch_size=self.batch_size,
                                dtype=tf.float32).clone(cell_state=decoder_initial_state)

            if decode == "training":

                baseline_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                baseline_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, baseline_helper, wrapper_state)

                baseline_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        baseline_decoder, maximum_iterations=self.decoder_max_length)

                self.baseline_sample_id = baseline_decoder_outputs.sample_id
                # self.baseline_sample_output = baseline_decoder_outputs.rnn_output


                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.decoder_inputs_length, time_major=True)

                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, train_helper, wrapper_state)

                train_decoder_outputs, train_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)


                if rl == True:
                    train_helper_2 = tf.contrib.seq2seq.SampleEmbeddingHelper(
                        self.embedding,
                        tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id, seed=seed)

                    train_decoder_2 = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, train_helper_2, wrapper_state)

                    train_decoder_outputs_2, _, _ = tf.contrib.seq2seq.dynamic_decode(
                            train_decoder_2, maximum_iterations=self.decoder_max_length)

                    self.train_sample_id = train_decoder_outputs_2.sample_id
                    # self.train_sample_output = train_decoder_outputs_2.rnn_output

                with tf.variable_scope('loss'):

                    if rl == True:
                        self.nll = tf.contrib.seq2seq.sequence_loss(
                            train_decoder_outputs_2.rnn_output,
                            self.train_sample_id,
                            self.target_weights[:, :tf.shape(self.train_sample_id)[1]],
                            average_across_timesteps=False,
                            average_across_batch=False
                        )
                        
                        self.nll = tf.reduce_sum(self.nll, 1)

                        self.rl_loss = tf.reduce_sum(tf.multiply(self.nll, (self.rewards - self.baseline)))\
                                     / tf.cast(self.batch_size, tf.float32)

                        self.covloss = self.target_weights * (tf.transpose(tf.squeeze(train_decoder_state.covloss.stack())))
                        self.covloss = tf.reduce_sum(self.covloss) / tf.cast(self.batch_size, tf.float32)

                        self.classify_loss = tf.contrib.seq2seq.sequence_loss(
                            train_decoder_outputs.rnn_output,
                            tf.transpose(self.decoder_targets),
                            self.target_weights,
                            average_across_timesteps=False,
                            average_across_batch=False
                        )

                        self.classify_loss = tf.reduce_sum(self.classify_loss) / tf.cast(self.batch_size, tf.float32)

                        self.non_rl_loss = self.classify_loss + self.coverage_lambda * self.covloss

                        self.loss = (1 - self.rl_lambda) * self.non_rl_loss + self.rl_lambda * self.rl_loss

                    else:

                        self.covloss = self.target_weights * (tf.transpose(tf.squeeze(train_decoder_state.covloss.stack())))
                        self.covloss = tf.reduce_sum(self.covloss) / tf.cast(self.batch_size, tf.float32)

                        self.classify_loss = tf.contrib.seq2seq.sequence_loss(
                            train_decoder_outputs.rnn_output,
                            tf.transpose(self.decoder_targets),
                            self.target_weights,
                            average_across_timesteps=False,
                            average_across_batch=False
                        )

                        self.classify_loss = tf.reduce_sum(self.classify_loss) / tf.cast(self.batch_size, tf.float32)

                        self.loss = self.classify_loss + self.coverage_lambda * self.covloss

                with tf.variable_scope('train_op'):
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

                    self.lr = tf.Variable(learning_rate, trainable=False)
                    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                    self.lr_update = tf.assign(self.lr, self.new_lr)

                    self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)
                    self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

            elif decode == "greedy":

                generation_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding,
                    tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)

                generation_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, generation_helper, wrapper_state)

                generation_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                    generation_decoder, maximum_iterations=decode_iter)

                self.translations = tf.expand_dims(generation_decoder_outputs.sample_id, -1)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})



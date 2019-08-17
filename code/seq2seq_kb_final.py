import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class Seq2seqKBFinalWrapperState(
    collections.namedtuple("Seq2seqKBFinalWrapperState", \
        ("cell_state", "attention_history", "pre_attention", "covloss", "time"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(Seq2seqKBFinalWrapperState, self)._replace(**kwargs))


class Seq2seqKBFinalWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, encoder_states, encoder_states_2, encoder_input_ids, encoder_input_2_ids, 
                    encoder_inputs_length, encoder_inputs_2_length, vocab_size,
                    attention_layer_size, source_hidden_size, target_hidden_size,
                    gen_vocab_size=None, encoder_state_size=None, initial_cell_state=None, name=None):
        
        super(Seq2seqKBFinalWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size

        self._encoder_input_ids = encoder_input_ids
        self._encoder_input_2_ids = encoder_input_2_ids
        self._encoder_states = encoder_states
        self._encoder_states_2 = encoder_states_2

        self._sen_len = tf.shape(self._encoder_input_ids)[1]

        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._encoder_inputs_length = encoder_inputs_length
        self._encoder_inputs_2_length = encoder_inputs_2_length
        
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size,
                            self._cell.output_size + self._encoder_state_size])
        self._copy_weight_2 = tf.get_variable('CopyWeight_2', [self._encoder_state_size,
                            self._cell.output_size + self._encoder_state_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")

        # attention weights
        self.Wh = tf.layers.Dense(attention_layer_size, use_bias=True, activation=None, name='Wh')
        self.Ws = tf.get_variable(name='Ws', shape=[target_hidden_size, attention_layer_size])
        self.wc = tf.get_variable(name='wc', shape=[1, attention_layer_size])
        self.v = tf.layers.Dense(1, use_bias=False, activation=None, name='v')

        self.Wh_2 = tf.layers.Dense(attention_layer_size, use_bias=True, activation=None, name='Wh_2')
        self.Ws_2 = tf.get_variable(name='Ws_2', shape=[target_hidden_size, attention_layer_size])
        self.Wa_2 = tf.get_variable(name='Wa_2', shape=[2 * source_hidden_size, attention_layer_size])
        self.v_2 = tf.layers.Dense(1, use_bias=False, activation=None, name='v_2')

        # fusion weights
        self.WF = tf.get_variable(name='WF', shape=[2 * source_hidden_size, 2 * source_hidden_size])
        self.WF_2 = tf.get_variable(name='WF_2', shape=[2 * source_hidden_size, 2 * source_hidden_size])
        self.F = tf.get_variable(name='F', shape=[2 * source_hidden_size, 1])
        self.F_2 = tf.get_variable(name='F_2', shape=[2 * source_hidden_size, 1])
        

    def __call__(self, inputs, state, scope=None):
        
        cell_state = state.cell_state
        attention_history = state.attention_history
        pre_attention = state.pre_attention

        outputs, cell_state = self._cell(tf.concat([inputs, pre_attention], 1), cell_state, scope)

        # attn1 = self.Wh(self._encoder_states) + tf.expand_dims(tf.matmul(outputs, self.Ws), 1)
        attn1 = self.Wh(self._encoder_states) + tf.expand_dims(tf.matmul(outputs, self.Ws), 1) + \
                tf.einsum("ijn,nk->ijk", tf.expand_dims(attention_history, 2), self.wc)
        attn2 = tf.squeeze(self.v(tf.tanh(attn1)), axis=[2]) #[batch_size,enc_seq]
        encoded_mask = (tf.sequence_mask(self._encoder_inputs_length,
                        dtype=tf.float32, name='encoded_mask') - 1) * 1e12
        attention_weight = tf.nn.softmax(attn2 + encoded_mask) #[batch_size,enc_seq]

        # attention_weight_new = tf.nn.softmax(attention_weight / tf.exp(attention_history))
        # context = tf.matmul(tf.expand_dims(attention_weight_new, 1), self._encoder_states)
        context = tf.matmul(tf.expand_dims(attention_weight, 1), self._encoder_states)
        context = tf.squeeze(context, [1])

        attn1_2 = self.Wh_2(self._encoder_states_2) + tf.expand_dims(tf.matmul(outputs, self.Ws_2), 1) + \
                tf.expand_dims(tf.matmul(context, self.Wa_2), 1)
        attn2_2 = tf.squeeze(self.v_2(tf.tanh(attn1_2)), axis=[2]) #[batch_size,enc_seq]
        encoded_mask_2 = (tf.sequence_mask(self._encoder_inputs_2_length,
                        dtype=tf.float32, name='encoded_mask_2') - 1) * 1e12
        attention_weight_2 = tf.nn.softmax(attn2_2 + encoded_mask_2) #[batch_size,enc_seq]

        context_2 = tf.matmul(tf.expand_dims(attention_weight_2, 1), self._encoder_states_2)
        context_2 = tf.squeeze(context_2, [1])

        # Gated Multimodal Unit
        h_c = tf.tanh(tf.matmul(context, self.WF))
        h_c_2 = tf.tanh(tf.matmul(context_2, self.WF_2))
        g_h = tf.nn.sigmoid(tf.matmul(context, self.F) + tf.matmul(context_2, self.F_2))
        h = g_h * h_c + (1 - g_h) * h_c_2

        # generator
        generate_score = self._projection(tf.concat([outputs, h], 1))
        g_score_total = tf.pad(tf.exp(generate_score), [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])

        # copy
        copy_score = tf.einsum("ijm,im->ij", tf.nn.tanh(tf.einsum("ijk,km->ijm",
                    self._encoder_states, self._copy_weight)), tf.concat([outputs, h], 1))
        encoded_mask_copy = tf.sequence_mask(self._encoder_inputs_length, dtype=tf.float32, name='encoded_mask_copy')
        copy_score = tf.exp(copy_score) * encoded_mask_copy
        batch_size = tf.shape(self._encoder_input_ids)[0]
        sen_len = tf.shape(self._encoder_input_ids)[1]
        batch_nums = tf.range(0, limit=batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        batch_nums = tf.tile(batch_nums, [1, sen_len]) # shape (batch_size, sen_len)
        indices = tf.stack((batch_nums, self._encoder_input_ids), axis=2) # shape (batch_size, sen_len, 2)
        shape = [batch_size, self._vocab_size]
        c_score_one_hot = tf.scatter_nd(indices, copy_score, shape)

        copy_score_2 = tf.einsum("ijm,im->ij", tf.nn.tanh(tf.einsum("ijk,km->ijm",
                    self._encoder_states_2, self._copy_weight_2)), tf.concat([outputs, h], 1))
        encoded_mask_copy_2 = tf.sequence_mask(self._encoder_inputs_2_length,
                                dtype=tf.float32, name='encoded_mask_copy_2')
        copy_score_2 = tf.exp(copy_score_2) * encoded_mask_copy_2
        batch_size_2 = tf.shape(self._encoder_input_2_ids)[0]
        sen_len_2 = tf.shape(self._encoder_input_2_ids)[1]
        batch_nums_2 = tf.range(0, limit=batch_size_2)
        batch_nums_2 = tf.expand_dims(batch_nums_2, 1)
        batch_nums_2 = tf.tile(batch_nums_2, [1, sen_len_2])
        indices_2 = tf.stack((batch_nums_2, self._encoder_input_2_ids), axis=2)
        shape_2 = [batch_size_2, self._vocab_size]
        c_score_one_hot_2 = tf.scatter_nd(indices_2, copy_score_2, shape_2)

        outputs_score = c_score_one_hot + c_score_one_hot_2 + g_score_total
        # outputs_score = g_score_total

        norm = tf.expand_dims(tf.reduce_sum(outputs_score, axis=1), 1)
        outputs = outputs_score / norm
        outputs = tf.log(outputs + 1e-12)

        covloss = tf.reduce_sum(tf.minimum(attention_history, attention_weight), 1, keepdims=True)
        pre_covloss = state.covloss
        cur_covloss = pre_covloss.write(state.time, covloss)

        state = Seq2seqKBFinalWrapperState(cell_state=cell_state,
            attention_history=attention_history+attention_weight,
            pre_attention=h,
            covloss=cur_covloss, time=state.time + 1)
        return outputs, state

    @property
    def state_size(self):
        return Seq2seqKBFinalWrapperState(cell_state=self._cell.state_size, 
                attention_history=self._sen_len, pre_attention=self._encoder_state_size, \
                covloss=1, time=tf.TensorShape([]))

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            attention_history = tf.zeros([batch_size, self._sen_len], tf.float32)
            pre_attention = tf.zeros([batch_size, self._encoder_state_size], tf.float32)
            covloss = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            return Seq2seqKBFinalWrapperState(cell_state=cell_state,
                    attention_history=attention_history, pre_attention=pre_attention, \
                    covloss=covloss, time=tf.zeros([], dtype=tf.int32))


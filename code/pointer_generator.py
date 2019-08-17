import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class PointerGeneratorWrapperState(
    collections.namedtuple("PointerGeneratorWrapperState", \
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
            super(PointerGeneratorWrapperState, self)._replace(**kwargs))


class PointerGeneratorWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, encoder_states, encoder_input_ids, encoder_inputs_length, vocab_size,
                    attention_layer_size, target_hidden_size,
                    gen_vocab_size=None, encoder_state_size=None, initial_cell_state=None, name=None):
        
        super(PointerGeneratorWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size

        self._encoder_input_ids = encoder_input_ids
        self._encoder_states = encoder_states

        self._sen_len = tf.shape(self._encoder_input_ids)[1]
        
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._encoder_inputs_length = encoder_inputs_length
        
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=True, name="OutputProjection")
        self.gen_prob_layer = tf.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, name="p_gen")

        #attention weights
        self.Wh = tf.layers.Dense(attention_layer_size, use_bias=True, activation=None)
        self.Ws = tf.get_variable(name='Ws', shape=[target_hidden_size, attention_layer_size])
        self.wc = tf.get_variable(name='wc', shape=[1, attention_layer_size])
        self.v = tf.layers.Dense(1, use_bias=False, activation=None)

    def __call__(self, inputs, state, scope=None):
        
        cell_state = state.cell_state
        attention_history = state.attention_history
        pre_attention = state.pre_attention

        outputs, cell_state = self._cell(tf.concat([inputs, pre_attention], 1), cell_state, scope)

        # attention
        attn1 = self.Wh(self._encoder_states) + tf.expand_dims(tf.matmul(outputs, self.Ws), 1) + \
                tf.einsum("ijn,nk->ijk", tf.expand_dims(attention_history, 2), self.wc)
        attn2 = tf.squeeze(self.v(tf.tanh(attn1)), axis=[2])
        encoded_mask = (tf.sequence_mask(self._encoder_inputs_length, dtype=tf.float32, name='encoded_mask') - 1) * 1e12
        attention_weight = tf.nn.softmax(attn2 + encoded_mask)

        context = tf.matmul(tf.expand_dims(attention_weight, 1), self._encoder_states)
        context = tf.squeeze(context, [1])

        # generator
        generate_score = self._projection(tf.concat([outputs, context], 1))
        prob_g = tf.nn.softmax(generate_score)
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])

        # pointer
        batch_size = tf.shape(self._encoder_input_ids)[0]
        sen_len = tf.shape(self._encoder_input_ids)[1]
        batch_nums = tf.range(0, limit=batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        batch_nums = tf.tile(batch_nums, [1, sen_len])
        indices = tf.stack((batch_nums, self._encoder_input_ids), axis=2)
        shape = [batch_size, self._vocab_size]
        prob_c_total = tf.scatter_nd(indices, attention_weight, shape)

        # generation probability
        p_gen = self.gen_prob_layer(tf.concat([inputs, outputs, context], 1))

        outputs = prob_g_total * p_gen + prob_c_total * (1 - p_gen)
        outputs = tf.log(outputs + 1e-12)

        covloss = tf.reduce_sum(tf.minimum(attention_history, attention_weight), 1, keepdims=True)
        pre_covloss = state.covloss
        cur_covloss = pre_covloss.write(state.time, covloss)

        state = PointerGeneratorWrapperState(cell_state=cell_state,
            attention_history=attention_history+attention_weight, pre_attention=context,
            covloss=cur_covloss, time=state.time + 1)
        return outputs, state

    @property
    def state_size(self):
        return PointerGeneratorWrapperState(cell_state=self._cell.state_size, 
                attention_history=self._sen_len, pre_attention=self._encoder_state_size, \
                covloss=1, time=tf.TensorShape([]))

    @property
    def output_size(self):
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
            return PointerGeneratorWrapperState(cell_state=cell_state,
                    attention_history=attention_history, pre_attention=pre_attention, \
                    covloss=covloss, time=tf.zeros([], dtype=tf.int32))


import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util

class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("cell_state", "last_ids", "prob_c"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))

class CopyNetWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, encoder_states, encoder_input_ids, encoder_inputs_length, vocab_size, copy,
                gen_vocab_size=None, encoder_state_size=None, initial_cell_state=None, name=None):
        
        super(CopyNetWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._copy = copy

        self._encoder_input_ids = encoder_input_ids
        self._encoder_inputs_length = encoder_inputs_length
        # self._encoder_input_ids = tf.transpose(encoder_input_ids, [1, 0])
        self._encoder_states = encoder_states
        # self._encoder_states = tf.transpose(encoder_states, [1, 0, 2])
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState."
                      "Received type %s instead."  % type(state))
        last_ids = state.last_ids
        prob_c = state.prob_c
        cell_state = state.cell_state

        encoded_mask = tf.sequence_mask(self._encoder_inputs_length, dtype=tf.float32, name='encoded_mask')

        # mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1),  self._encoder_input_ids), tf.float32)
        # mask = mask * encoded_mask
        # mask_prob_c = mask * prob_c
        # rou = mask_prob_c
        # selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        # inputs = tf.concat([inputs, selective_read], 1)

        outputs, cell_state = self._cell(inputs, cell_state, scope)

        # generation
        generate_score = self._projection(outputs)
        g_score_total = tf.pad(tf.exp(generate_score), [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])

        # copy
        copy_score = tf.einsum("ijm,im->ij", tf.nn.tanh(tf.einsum("ijk,km->ijm",
                    self._encoder_states, self._copy_weight)), outputs)
        copy_score = tf.exp(copy_score) * encoded_mask
        batch_size = tf.shape(self._encoder_input_ids)[0]
        sen_len = tf.shape(self._encoder_input_ids)[1]
        batch_nums = tf.range(0, limit=batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        batch_nums = tf.tile(batch_nums, [1, sen_len])
        indices = tf.stack((batch_nums, self._encoder_input_ids), axis=2)
        shape = [batch_size, self._vocab_size]
        c_score_one_hot = tf.scatter_nd(indices, copy_score, shape)

        if self._copy == True:
            outputs_score = c_score_one_hot + g_score_total
        else:
            outputs_score = g_score_total

        norm = tf.expand_dims(tf.reduce_sum(outputs_score, axis=1), 1)
        outputs = outputs_score / norm
        outputs = tf.log(outputs + 1e-12)

        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        prob_c = copy_score / norm
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)

        return outputs, state

    @property
    def state_size(self):
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = self._encoder_state_size)

    @property
    def output_size(self):
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.ones([batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            prob_c = prob_c / tf.expand_dims(tf.reduce_sum(prob_c, axis=1), 1)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)

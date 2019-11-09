import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """
    calculate encoded output and hidden state from encoder input and initialized encoder hidden
    batch by batch and paragraph by paragraph
    input is [batch_sz,max_train_x] encoder hidden is [batch_sz, enc_units]
    output is [batch_sz,max_train_x,enc_units]
    """
    def __init__(self, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.batch_sz = batch_sz
        self.enc_units = enc_units // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bidirectional_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, x, hidden):
        x = self.embedding(x)
        # [batch_sz,max_train_x,embedding_dim]
        output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[hidden, hidden])
        enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state],axis=-1)
        # output is [batch_sz, max_train_x, enc_units] enc_hidden after concat is [batch_sz, enc_units]
        return output, enc_hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """
    calculate attention and coverage from dec_hidden enc_output and prev_coverage
    one dec_hidden(word) by one dec_hidden
    dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
    prev_coverage is [batch_sz, max_train_x, 1]
    dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
    output context_vector [batch_sz, enc_units] attention_weights [batch_sz, max_train_x, 1] coverage [batch_sz, max_train_x, 1]

    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_h = tf.keras.layers.Dense(units)
        self.W_s = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_pad_mask, use_coverage, prev_coverage):

        # prev_coverage [batch_sz, max_len, 1]
        # 11.07 add enc_pad_mask [batch_sz, max_len_x] to mask attention
        # query or dec_hidden [batch_sz, enc_units], values or enc_output [batch_sz, max_len, enc_units]
        # hidden_with_time_axis shape == (batch_size, 1, enc_units)
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)
            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)

            coverage = attention_weights + prev_coverage

        else:
            # self.W1(values) [batch_sz, max_len, units] self.W2(hidden_with_time_axis): [batch_sz, 1, units]
            # score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))
            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            #attention_weights = masked_attention(attention_weights)
            if use_coverage:
                coverage = attention_weights

        # [batch_sz, max_len, enc_units]
        context_vector = attention_weights * enc_output
        # [batch_sz, enc_units]
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights, coverage


class Decoder(tf.keras.layers.Layer):
    """
    calculate output before pointer generator network
    input dec_inp [batch_sz, 1], hidden [batch_sz, enc_units], context_vector [batch_sz, enc_units]
    output dec_inp_context [batch_sz,1,embedding_dim+enc_units] dec_pred [batch_size,vocab_size] dec_hidden [batch_sz,dec_units]
    """
    def __init__(self, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, dec_inp, dec_hidden, context_vector):
        # context_vector[batch_sz, enc_units]
        # dec_hidden [batch_sz, dec_units] NOTE dec_units == enc_units
        # dec_inp shape [batch_size, 1, embedding_dim]
        dec_inp = self.embedding(dec_inp)
        # dec_inp shape [batch_sz, 1, embedding_dim + enc_units]
        dec_inp_context = tf.concat([tf.expand_dims(context_vector, 1), dec_inp], axis=-1)
        # output [batch_sz, 1, dec_units] state [batch_sz, dec_units]
        output, dec_hidden = self.gru(dec_inp_context) #, initial_state=dec_hidden)
        # output shape [batch_size, dec_units]
        output = tf.reshape(output, (-1, output.shape[2]))
        # print('output deduced by dec_hidden', tf.math.reduce_sum(output-dec_hidden)) they are same!!!
        # dec_inp shape [batch_size, vocab_size]
        dec_pred = self.fc(output)
        return dec_inp_context, dec_pred, dec_hidden


class Pointer(tf.keras.layers.Layer):
    """
    calculate Pgen
    input context_vector [batch_sz,enc_units] dec_hidden [batch_sz,dec_units] dec_inp_context [batch_sz,1,embedding_dim+enc_units]
    output scaler pgen
    """
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        # change dec_inp_context to [batch_sz,embedding_dim+enc_units]
        dec_inp = tf.squeeze(dec_inp, axis=1)
        pgen = tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
        return pgen

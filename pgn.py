<<<<<<< HEAD
import tensorflow as tf
from model_layers import Encoder, BahdanauAttention, Decoder, Pointer


class PGN(tf.keras.Model):
    """
    create pgn model
    input
    output
    """
    def __init__(self, gru_units, att_units, batch_sz, embedding_matrix):
        super(PGN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.enc_units = gru_units
        self.dec_units = gru_units
        self.att_units = att_units
        self.encoder = Encoder(self.enc_units, batch_sz, embedding_matrix)
        self.attention = BahdanauAttention(self.att_units)
        self.decoder = Decoder(self.dec_units, batch_sz, embedding_matrix)
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()    # [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)   # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        return enc_output, enc_hidden

    # NOTE dec_inp [batch_sz, max_train_y]
    def call(self, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []
        # initiate_hidden and context_vector for pgn
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)  # same as above
        # initiate coverage to None or whatever was passed in
        coverage_ret = prev_coverage
        dec_hidden = enc_hidden
        # initiate context_vector and coverage_ret
        context_vector, attention_weights, coverage_ret = self.attention(dec_hidden,
                                                                         enc_output,
                                                                         use_coverage=use_coverage,
                                                                         prev_coverage=coverage_ret)

        for t in range(dec_inp.shape[1]):  # iterate over time step max_len_y 33

            # attentions.append(attention_weights)
            # decoder takes dec_inp, dec_hidden, context_vector, dec_inp shape [batch_size, 1]
            # dec_hidden [batch_sz, dec_units] NOTE dec_units == enc_units, context_vector[batch_sz, enc_units]
            # decoder gives dec_pred [batch_size,vocab_size] dec_hidden [batch_sz,dec_units]
            # using teaching force!!!
            dec_inp_context, dec_pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1), dec_hidden, context_vector)
            # attention takes dec_hidden, enc_output, use_coverage = True, prev_coverage = None
            # attention gives context_vector, attention_weights, coverage
            context_vector, attention_weights, coverage_ret = self.attention(dec_hidden, enc_output,
                                                                             use_coverage=use_coverage,
                                                                             prev_coverage=coverage_ret)
            # pointer takes context_vector, dec_hidden, dec_inp
            pgen = self.pointer(context_vector, dec_hidden, dec_inp_context)
            attentions.append(attention_weights)
            coverages.append(coverage_ret)
            predictions.append(dec_pred)
            p_gens.append(pgen)

        # enc_extended_input [batch_sz, max_len_x] predictions [max_len_y, batch_sz, vocab_size]
        # attentions [max_len_y, batch_sz, max_len_x, 1] p_gens [max_len_y,]
        final_dist = self._calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len)
        # change shape of final_dist from [max_len_y, batch_sz, extend_vocab_size]
        # to [batch_sz, max_len_y, extend_vocab_size]
        final_dist = tf.stack(final_dist, 1)
        # to [batch_sz, max_len_y]
        #final_dist = tf.math.argmax(final_dist, axis=-1)

        return final_dist, attentions, coverages


    def _calc_final_dist(self, _enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len):
        """
        Calculate the final distribution, for the pointer-generator model
        Args:
        vocab_dists, prediction of decoder List length max_dec_steps of (batch_sz, vocab_size) array.
                    The words are in the order they appear in the vocabulary file.
        attn_dists: The attention distributions. List length max_dec_steps of (batch_size, max_train_x, 1) array.
        _enc_batch_extend_vocab, tokenized enc input (batch_sz, max_train_x) with pgn the in-article oov word is
        tonkenized with extended index.
        Returns:
        final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vocab_size) arrays.
        """
        batch_sz, vocab_size = vocab_dists[0].shape[0], vocab_dists[0].shape[1]
        attn_dists = tf.squeeze(attn_dists, axis = -1) # change to max_dec_steps of (batch_size, max_train_x) array
        batch_oov_len = tf.reduce_max(batch_oov_len)  # the maximum (over the batch) size of the extended vocabulary

        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vocab_size = vocab_size + batch_oov_len
        extra_zeros = tf.zeros((batch_sz, batch_oov_len))
        # list length max_dec_steps of shape (batch_size, extended_vsize) [max_len_y, batch_sz, extended_vsize]
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
        # then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_sz)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        max_train_x = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, max_train_x])  # shape (batch_size, max_train_x)
        indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, max_train_x, 2)
        shape = [batch_sz, extended_vocab_size]
        # list length max_dec_steps (batch_size, extended_vocab_size)
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
        # the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist+copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended,attn_dists_projected)]

        return final_dists

=======
import tensorflow as tf
from model_layers import Encoder, BahdanauAttention, Decoder, Pointer


class PGN(tf.keras.Model):
    """
    create pgn model
    input
    output
    """
    def __init__(self, gru_units, att_units, batch_sz, embedding_matrix):
        super(PGN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.enc_units = gru_units
        self.dec_units = gru_units
        self.att_units = att_units
        self.encoder = Encoder(self.enc_units, batch_sz, embedding_matrix)
        self.attention = BahdanauAttention(self.att_units)
        self.decoder = Decoder(self.dec_units, batch_sz, embedding_matrix)
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()    # [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)   # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        return enc_output, enc_hidden

    # NOTE dec_inp [batch_sz, max_train_y]
    def call(self, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []
        # initiate_hidden and context_vector for pgn
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)  # same as above
        # initiate coverage to None or whatever was passed in
        coverage_ret = prev_coverage
        dec_hidden = enc_hidden
        # initiate context_vector and coverage_ret
        context_vector, attention_weights, coverage_ret = self.attention(dec_hidden,
                                                                         enc_output,
                                                                         use_coverage=use_coverage,
                                                                         prev_coverage=coverage_ret)

        for t in range(dec_inp.shape[1]):  # iterate over time step max_len_y 33

            # attentions.append(attention_weights)
            # decoder takes dec_inp, dec_hidden, context_vector, dec_inp shape [batch_size, 1]
            # dec_hidden [batch_sz, dec_units] NOTE dec_units == enc_units, context_vector[batch_sz, enc_units]
            # decoder gives dec_pred [batch_size,vocab_size] dec_hidden [batch_sz,dec_units]
            # using teaching force!!!
            dec_inp_context, dec_pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1), dec_hidden, context_vector)
            # attention takes dec_hidden, enc_output, use_coverage = True, prev_coverage = None
            # attention gives context_vector, attention_weights, coverage
            context_vector, attention_weights, coverage_ret = self.attention(dec_hidden, enc_output,
                                                                             use_coverage=use_coverage,
                                                                             prev_coverage=coverage_ret)
            # pointer takes context_vector, dec_hidden, dec_inp
            pgen = self.pointer(context_vector, dec_hidden, dec_inp_context)
            attentions.append(attention_weights)
            coverages.append(coverage_ret)
            predictions.append(dec_pred)
            p_gens.append(pgen)

        # enc_extended_input [batch_sz, max_len_x] predictions [max_len_y, batch_sz, vocab_size]
        # attentions [max_len_y, batch_sz, max_len_x, 1] p_gens [max_len_y,]
        final_dist = self._calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len)
        # change shape of final_dist from [max_len_y, batch_sz, extend_vocab_size]
        # to [batch_sz, max_len_y, extend_vocab_size]
        final_dist = tf.stack(final_dist, 1)
        # to [batch_sz, max_len_y]
        #final_dist = tf.math.argmax(final_dist, axis=-1)

        return final_dist, attentions, coverages


    def _calc_final_dist(self, _enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len):
        """
        Calculate the final distribution, for the pointer-generator model
        Args:
        vocab_dists, prediction of decoder List length max_dec_steps of (batch_sz, vocab_size) array.
                    The words are in the order they appear in the vocabulary file.
        attn_dists: The attention distributions. List length max_dec_steps of (batch_size, max_train_x, 1) array.
        _enc_batch_extend_vocab, tokenized enc input (batch_sz, max_train_x) with pgn the in-article oov word is
        tonkenized with extended index.
        Returns:
        final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vocab_size) arrays.
        """
        batch_sz, vocab_size = vocab_dists[0].shape[0], vocab_dists[0].shape[1]
        attn_dists = tf.squeeze(attn_dists, axis = -1) # change to max_dec_steps of (batch_size, max_train_x) array
        batch_oov_len = tf.reduce_max(batch_oov_len)  # the maximum (over the batch) size of the extended vocabulary

        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vocab_size = vocab_size + batch_oov_len
        extra_zeros = tf.zeros((batch_sz, batch_oov_len))
        # list length max_dec_steps of shape (batch_size, extended_vsize) [max_len_y, batch_sz, extended_vsize]
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
        # then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_sz)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        max_train_x = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, max_train_x])  # shape (batch_size, max_train_x)
        indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, max_train_x, 2)
        shape = [batch_sz, extended_vocab_size]
        # list length max_dec_steps (batch_size, extended_vocab_size)
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
        # the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist+copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended,attn_dists_projected)]

        return final_dists

>>>>>>> a084f9ce23303f1bf8968fa732051cb08c426656

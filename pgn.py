import tensorflow as tf
import numpy as np
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
    def call(self, dec_inp, enc_extended_inp, enc_pad_mask, batch_oov_len, enc_output, dec_hidden,
             use_coverage=True, prev_coverage=None, prediction=False):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []

        # # initiate_hidden and context_vector for pgn
        # enc_hidden = self.encoder.initialize_hidden_state()
        # enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)  # same as above
        # # initiate coverage to None or whatever was passed in
        # dec_hidden = enc_hidden
        # # initiate context_vector and coverage_ret

        context_vector, attention_weights, coverage_ret = self.attention(dec_hidden,enc_output,enc_pad_mask,use_coverage,prev_coverage)

        if prediction:
            decode_steps = dec_inp.shape[1]
        else:
            decode_steps = dec_inp.shape[1] - 1

        for t in range(decode_steps):  # 11.11 iterate over time step max_len_y - 1 !!!!

            # attentions.append(attention_weights)
            # decoder takes dec_inp, dec_hidden, context_vector, dec_inp shape [batch_size, 1]
            # dec_hidden [batch_sz, dec_units] NOTE dec_units == enc_units, context_vector[batch_sz, enc_units]
            # decoder gives dec_pred [batch_size,vocab_size] dec_hidden [batch_sz,dec_units]
            # using teaching force!!!
            attentions.append(attention_weights)
            coverages.append(coverage_ret)
            dec_inp_context, dec_pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1), dec_hidden, context_vector)
            # attention takes dec_hidden, enc_output, use_coverage = True, prev_coverage = None
            # attention gives context_vector, attention_weights, coverage
            if not prediction:
                context_vector, attention_weights, coverage_ret = self.attention(dec_hidden, enc_output,enc_pad_mask,use_coverage,coverage_ret)
            # pointer takes context_vector, dec_hidden, dec_inp
            pgen = self.pointer(context_vector, dec_hidden, dec_inp_context)
            predictions.append(dec_pred)
            p_gens.append(pgen)


        # enc_extended_input [batch_sz, max_len_x] predictions [max_len_y, batch_sz, vocab_size]
        # attentions [max_len_y, batch_sz, max_len_x, 1] p_gens [max_len_y,]
        final_dist = self._calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len)
        # change shape of final_dist from [max_len_y, batch_sz, extend_vocab_size]
        # to [batch_sz, max_len_y, extend_vocab_size]
        #final_dist = tf.transpose(final_dist, [1, 0, 2])

        return final_dist, attentions, coverages, dec_hidden, context_vector, p_gens


    def _calc_final_dist(self, enc_extended_inp, vocab_dists, attn_dists, p_gens, batch_oov_len):
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

        max_len_y, batch_sz, vocab_size = len(vocab_dists), vocab_dists[0].shape[0], vocab_dists[0].shape[1]
        attn_dists = tf.squeeze(attn_dists, axis = -1) # change to max_dec_steps of (batch_size, max_train_x) array
        batch_oov_len = tf.reduce_max(batch_oov_len)  # the maximum (over the batch) size of the extended vocabulary

        # p_gens = tf.convert_to_tensor(p_gens)
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]
        # to substitute above code
        # def weight_cross(p_gens, vocab_dists):
        #     list_like = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        #     for i in range(len(p_gens)):
        #         list_like = list_like.write(i, p_gens[i] * vocab_dists[i])
        #     return list_like.stack()
        # vocab_dists = weight_cross(p_gens, vocab_dists)
        # attn_dists = weight_cross((1-p_gens), attn_dists)

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        extended_vocab_size = vocab_size + batch_oov_len
        extra_zeros = tf.zeros((batch_sz, batch_oov_len))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # extra_zeros = tf.zeros((max_len_y, batch_sz, batch_oov_len))
        # vocab_dists_extended = tf.concat(axis=2, values=[vocab_dists,extra_zeros])
        # list length max_dec_steps of shape (batch_size, extended_vsize) [max_len_y, batch_sz, extended_vsize]
        #vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]
        # to substitute above code


        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
        # then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_sz)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        max_len_x = tf.shape(enc_extended_inp)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, max_len_x])  # shape (batch_size, max_train_x)
        indices = tf.stack((batch_nums, enc_extended_inp), axis=2)  # shape (batch_size, max_train_x, 2)
        shape = (batch_sz, extended_vocab_size)
        # list length max_dec_steps (batch_size, extended_vocab_size)
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]
        # substitute above code
        # temp = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False)
        # for i in range(p_gens.shape[0]):  # which equal to max_len_y
        #     temp = temp.write(i, tf.scatter_nd(indices, attn_dists[i], shape))
        # attn_dists_projected = temp.stack()

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
        # the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [vocab_dist+copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended,attn_dists_projected)]
        # to substitute code above
        # final_dists = tf.add(vocab_dists_extended,attn_dists_projected)

        return final_dists


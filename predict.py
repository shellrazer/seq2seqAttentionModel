import tensorflow as tf
from pgn import PGN
from data_loader import batch, get_token, token_to_word
import numpy as np
from gensim.models import Word2Vec
import os
import time




#@tf.function
# decode for one batch or one beam_size
def beam_decode(w2v_model, max_len_y, min_dec_length, beam_size, enc_inp, enc_extended_inp, enc_pad_mask, batch_oov_len, enc_oov_dict):


    def decode_onestep(dec_inp, dec_hidden, enc_output, enc_extended_inp,enc_pad_mask,batch_oov_len,coverage_ret=None):
        # dec_inp prediction from last step [beam_size,1], enc_extended_inp, enc_pad_mask [beam_size,max_len_x]
        # batch_oov_len [beam_size,] enc_output [beam_size, max_len_x, enc_units]
        # dec_hidden from last step [beam_size, dec_units] coverage_ret [beam_size, max_len_x, 1]
        # final_dist [[beam_size, extend_vocab_size]]  attentions, coverages [[beam_size, max_len_x, 1]]

        final_dist, attentions, coverages, dec_hidden, context_vector, p_gens = model(dec_inp, enc_extended_inp,
                                                                                      enc_pad_mask, batch_oov_len,
                                                                                      enc_output, dec_hidden,
                                                                                      use_coverage=True, prev_coverage=coverage_ret,
                                                                                      prediction=True)
        att_dist = attentions[0]  # [batch_sz, max_len_x, 1]
        coverage_ret = coverages[0]  # [batch_sz, max_len_x, 1]
        p_gen = p_gens[0]  # scaler
        final_dist = final_dist[0]   # [batch_sz, extend_vocab_size]
        top_k_probs, top_k_ids = tf.nn.top_k(final_dist, k=beam_size)
        top_k_log_probs = tf.math.log(top_k_probs)
        results = {"last_context_vector": context_vector,
                   "dec_hidden": dec_hidden,
                   "attention_vec": att_dist,
                   "coverage":coverage_ret,
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "p_gen": p_gen}
        return results

    # Class designed to hold hypothesises throughout the beamSearch decoding
    class hypothesis:
        def __init__(self, tokens, log_probs, dec_hidden, attn_dists, coverage_ret, p_gens):
            self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
            self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
            self.dec_hidden = dec_hidden  # decoder state after the last token decoding
            self.attn_dists = attn_dists  # attention dists of all the tokens
            self.coverage_ret = coverage_ret
            self.p_gens = p_gens  # generation probability of all the tokens
            self.abstract = ""
            self.text = ""
            self.real_abstract = ""

        def extend(self, token, log_prob, dec_hidden, attn_dist, coverage_ret, p_gen):
            """Method to extend the current hypothesis by adding the next decoded toekn and all the informations associated with it"""
            return hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                              log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                              dec_hidden=dec_hidden,  # we update the state
                              attn_dists=self.attn_dists + [attn_dist],
                              coverage_ret=coverage_ret,  # update coverage_ret
                              # we add the attention dist and coverage of the decoded token
                              p_gens=self.p_gens + [p_gen])
        @property
        def latest_token(self):
            return self.tokens[-1]
        @property
        def tot_log_prob(self):
            return sum(self.log_probs)
        @property
        def avg_log_prob(self):
            return self.tot_log_prob / len(self.tokens)

    START_index = w2v_model.wv.vocab['<START>'].index
    STOP_index = w2v_model.wv.vocab['<STOP>'].index
    PAD_index = w2v_model.wv.vocab['<PAD>'].index
    UNK_index = w2v_model.wv.vocab['<UNK>'].index
    vocab_size = len(w2v_model.wv.vocab)
    enc_inp = tf.tile(enc_inp,[beam_size,1])
    enc_extended_inp = tf.tile(enc_extended_inp,[beam_size,1])
    enc_pad_mask = tf.tile(enc_pad_mask,[beam_size,1])
    batch_oov_len = tf.tile(batch_oov_len,[beam_size,])
    enc_oov_dict = [enc_oov_dict for _ in range(beam_size)]
    enc_output, enc_hidden = model.call_encoder(enc_inp) # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
    dec_hidden = enc_hidden


    # end of the nested class
    # Initial Hypothesises (beam_size many list)
    hyps = [hypothesis(tokens=[START_index],
                       # we initalize all the beam_size hypothesises with the token start
                       log_probs=[0.0],  # Initial log prob = 0
                       dec_hidden=dec_hidden[0],
                       # initial dec_state (we will use only the first dec_state because they're initially the same)
                       attn_dists=tf.zeros([max_len_x, 1],dtype=tf.dtypes.float32),
                       coverage_ret=tf.zeros([max_len_x, 1],dtype=tf.dtypes.float32),
                       p_gens=[],  # we init the coverage vector to zero
                       ) for _ in range(beam_size)]  # batch_size == beam_size

    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step

    while steps < max_len_y and len(results) < beam_size:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        latest_tokens = [t if t in range(vocab_size) else UNK_index for t in latest_tokens]  # we replace all the oov is by the unknown token
        dec_hidden = [h.dec_hidden for h in hyps]  # we collect the last states for each hypothesis
        coverage_ret = [h.coverage_ret for h in hyps]

        # we decode the top likely beam_size tokens tokens at time step t for each hypothesis
        # decode_onestep(dec_inp, dec_hidden, enc_output, enc_extended_inp,enc_pad_mask,batch_oov_len,coverage_ret=None)
        returns = decode_onestep(tf.expand_dims(latest_tokens, axis=1), tf.stack(dec_hidden, axis=0), enc_output,
                                 enc_extended_inp, enc_pad_mask, batch_oov_len, coverage_ret=tf.stack(coverage_ret, axis=0))
        topk_ids, topk_log_probs, new_dec_hiddens, attn_dists, new_coverage_rets, p_gens = returns['top_k_ids'], \
                                                                                         returns['top_k_log_probs'], \
                                                                                         returns['dec_hidden'], \
                                                                                         returns['attention_vec'], \
                                                                                         returns['coverage'], \
                                                                                         returns["p_gen"]
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        for i in range(num_orig_hyps):
            h, new_dec_hidden, attn_dist, new_coverage_ret, p_gen = hyps[i], new_dec_hiddens[i], attn_dists[i], new_coverage_rets[i], p_gens[i]

            for j in range(beam_size):
                # we extend each hypothesis with each of the top k tokens (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j],
                                   dec_hidden=new_dec_hidden,
                                   attn_dist=attn_dist,
                                   coverage_ret=new_coverage_ret,
                                   p_gen=p_gen)
                all_hyps.append(new_hyp)

        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == STOP_index:
                if steps >= min_dec_length:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence, given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    best_hyp.abstract = token_to_word(w2v_model, best_hyp.tokens, [])
    print(best_hyp.abstract)
    best_hyp.text = token_to_word(w2v_model, enc_inp[0], enc_oov_dict)
    print(best_hyp.text)
    return best_hyp


#########################################start here#########################################################
w2v_model = Word2Vec.load('./word2vec.model')
print('w2v model loaded')
max_len_x = 103
max_len_y = 40
min_len_y = 5

embedding_matrix = np.loadtxt('embedding_matrix.txt', dtype=np.float32)
print('embedding_matrix loaded')
beam_size = 3
batch_sz = beam_size

test_X = []
test_X_oov = []

with open('./data/test_X_pad.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        test_X.append(line.strip().split(' '))
f.close()
with open('./data/test_X_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        test_X_oov.append(line.strip().split(' '))
f.close()


# dataset_token, dataset_extended_token, dataset_pad_mask, dataset_oov_dict, dataset_oov_len
test_X_token, test_X_extended_token, test_X_pad_mask, test_X_oov_dict, test_X_oov_len = get_token(w2v_model, max_len_x, test_X, test_X_oov)
test_X_token, test_X_extended_token, test_X_pad_mask, test_X_oov_len = tf.convert_to_tensor(test_X_token), \
                                                                       tf.convert_to_tensor(test_X_extended_token), \
                                                                       tf.convert_to_tensor(test_X_pad_mask), \
                                                                       tf.convert_to_tensor(test_X_oov_len)
dataset = tf.data.Dataset.from_tensor_slices((test_X_token, test_X_extended_token, test_X_pad_mask, test_X_oov_len))
dataset_batch = dataset.batch(batch_size=1, drop_remainder=True)
dataset_len = len(test_X_token)
dataset_oov_dict = test_X_oov_dict

gru_units = 512
att_units = 64
embedding_matrix = embedding_matrix

model = PGN(gru_units, att_units, batch_sz, embedding_matrix)

optimizer = tf.keras.optimizers.Adam(clipvalue=2.0)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=model.encoder, attention=model.attention,
                                 decoder=model.decoder, pointer=model.pointer)

status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# status.assert_consumed()
print('model restored')

res=[]
for (batch, (enc, enc_extend, enc_mask, enc_oov_len)) in enumerate(dataset_batch.take(dataset_len)):
    enc_oov_dict = dataset_oov_dict[batch:(batch + 1)]
    print('decode sample {}'.format(batch+1))
    ans = beam_decode(w2v_model, max_len_y, min_len_y, beam_size, enc, enc_extend, enc_mask, enc_oov_len, enc_oov_dict)
    res.append([ans.text, ans.abstract])
with open('./test_results.txt', 'w', encoding='utf-8') as f:
    for line in res:
        line = '|'.join(line)
        f.write(line)
        f.write('\n')
    print('test results saved')

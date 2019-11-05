<<<<<<< HEAD
import tensorflow as tf
from pgn import PGN
from data_loader import batch, get_token, token_to_word
import numpy as np
from gensim.models import Word2Vec
import os
import time


def loss_function(real, pred, padding_mask):
    #  pred & real & mask [batch_sz, max_len_y]
    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = 0
    batch_sz, max_len_y = pred.shape[0], pred.shape[1]
    for t in range(max_len_y):
        loss_ = loss_object(real[:,t], pred[:,t])
        mask = tf.cast(padding_mask[:,t], dtype=loss_.dtype)
        loss_ *= mask
        # print('loss_:', loss_)
        loss_ = tf.reduce_mean(loss_)
        loss += loss_
    return loss


def coverage_loss(attn_dists, coverages, padding_mask):
    """
    Calculates the coverage loss from the attention distributions.
      Args:
        attn_dists coverages: [max_len_y, batch_sz, max_len_x, 1]
        padding_mask: shape (batch_size, max_len_y).
      Returns:
        coverage_loss: scalar
    """
    covlosses = []
    # transfer attn_dists coverages to [max_len_y, batch_sz, max_len_x]
    attn_dists = tf.squeeze(attn_dists, axis=-1)
    coverages = tf.squeeze(coverages, axis=-1)
    max_len_y = attn_dists.shape[0]
    for t in range(max_len_y):
        covloss_ = tf.reduce_sum(tf.minimum(attn_dists[t,:,:],coverages[t,:,:]), axis=1)
        covlosses.append(covloss_)
    covlosses = tf.stack(covlosses, 1)  # change from[max_len_y, batch_sz] to [batch_sz, max_len_y]
    mask = tf.cast(padding_mask, dtype=covloss_.dtype)
    covlosses *= mask  #covloss [batch_sz, max_len_y]
    loss = tf.reduce_sum(tf.reduce_mean(covlosses,axis=0))   # mean loss of each time step and then sum up
    return loss


def train_one_step(model, inp, targ, enc_extended_inp, batch_oov_len, cov_loss_wt = 1.0, padding_mask=None):
    loss = 0
    with tf.GradientTape() as tape:
        # enc_output, enc_hidden = model.call_encoder(inp) # is this necessary? !!!
        # call(self, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None)
        # final_dist [batch_sz, max_len_y, extend_vocab_size] attentions, coverages [max_len_y, batch_sz, max_len_x, 1]
        final_dist, attentions, coverages = model(inp, targ, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None)
        loss = loss_function(targ, final_dist, padding_mask) + cov_loss_wt * coverage_loss(attentions, coverages, padding_mask)
        batch_loss = (loss / int(targ.shape[1]))
    variables = model.trainable_variables
    # print(variables)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss




w2v_model = Word2Vec.load('./word2vec.model')
max_lens = [243, 244, 33, 254]
#embedding_matrix = get_embedding_matrix(w2v_model)
embedding_matrix = np.loadtxt('embedding_matrix.txt', dtype=np.float32)
batch_sz = 64

train_X = []
train_X_oov = []
train_y = []
train_y_oov = []
with open('./data/train_X_pad.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_X.append(line.strip().split(' '))
f.close()
with open('./data/train_X_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_X_oov.append(line.strip().split(' '))
f.close()
with open('./data/train_y_pad.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_y.append(line.strip().split(' '))
f.close()
with open('./data/train_y_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_y_oov.append(line.strip().split(' '))
f.close()

train_X_token, train_X_extended_token, train_X_pad_mask, train_X_oov_dict, train_X_oov_len = get_token(w2v_model, max_lens[1], train_X, train_X_oov)
train_y_token, _, train_y_pad_mask, _, _ = get_token(w2v_model, max_lens[2], train_y)

dataset_train_batch, dataset_test_batch = batch(64, train_X_token, train_X_extended_token, train_X_pad_mask, train_X_oov_len, train_y_token, train_y_pad_mask)
example_input_batch, example_enc_extend, example_enc_mask,example_oov_len, example_target_batch, example_target_mask = next(iter(dataset_train_batch))

gru_units = 256
att_units = 50
embedding_matrix = embedding_matrix

pgn_model = PGN(gru_units, att_units, batch_sz, embedding_matrix)
# final_dist, attentions, coverages = pgn_model(example_input_batch, example_target_batch, example_enc_extend, example_oov_len, use_coverage=True, prev_coverage=None)
print('finish')

optimizer = tf.keras.optimizers.Adam(clipvalue=2.0)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# loss_one_step = train_one_step(pgn_model, example_input_batch, example_target_batch, example_enc_extend, example_oov_len, cov_loss_wt = 1.0, padding_mask=example_target_mask)
# print(loss_one_step)

EPOCHS = 5
steps_per_epoch = 70761 // batch_sz #!!!
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint()

for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    for (batch, (example_input_batch, example_enc_extend, example_enc_mask,example_oov_len, example_target_batch,
                 example_target_mask)) in enumerate(dataset_test_batch.take(steps_per_epoch)):
        batch_loss = train_one_step(pgn_model, example_input_batch, example_target_batch, example_enc_extend,
                                    example_oov_len, cov_loss_wt = 1.0, padding_mask=example_target_mask)
        total_loss += batch_loss

        if batch % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
=======
import tensorflow as tf
from pgn import PGN
from data_loader import batch, get_token, token_to_word
import numpy as np
from gensim.models import Word2Vec
import os
import time


def loss_function(real, pred, padding_mask):
    #  pred & real & mask [batch_sz, max_len_y]
    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = 0
    batch_sz, max_len_y = pred.shape[0], pred.shape[1]
    for t in range(max_len_y):
        loss_ = loss_object(real[:,t], pred[:,t])
        mask = tf.cast(padding_mask[:,t], dtype=loss_.dtype)
        loss_ *= mask
        # print('loss_:', loss_)
        loss_ = tf.reduce_mean(loss_)
        loss += loss_
    return loss


def coverage_loss(attn_dists, coverages, padding_mask):
    """
    Calculates the coverage loss from the attention distributions.
      Args:
        attn_dists coverages: [max_len_y, batch_sz, max_len_x, 1]
        padding_mask: shape (batch_size, max_len_y).
      Returns:
        coverage_loss: scalar
    """
    covlosses = []
    # transfer attn_dists coverages to [max_len_y, batch_sz, max_len_x]
    attn_dists = tf.squeeze(attn_dists, axis=-1)
    coverages = tf.squeeze(coverages, axis=-1)
    max_len_y = attn_dists.shape[0]
    for t in range(max_len_y):
        covloss_ = tf.reduce_sum(tf.minimum(attn_dists[t,:,:],coverages[t,:,:]), axis=1)
        covlosses.append(covloss_)
    covlosses = tf.stack(covlosses, 1)  # change from[max_len_y, batch_sz] to [batch_sz, max_len_y]
    mask = tf.cast(padding_mask, dtype=covloss_.dtype)
    covlosses *= mask  #covloss [batch_sz, max_len_y]
    loss = tf.reduce_sum(tf.reduce_mean(covlosses,axis=0))   # mean loss of each time step and then sum up
    return loss


def train_one_step(model, inp, targ, enc_extended_inp, batch_oov_len, cov_loss_wt = 1.0, padding_mask=None):
    loss = 0
    with tf.GradientTape() as tape:
        # enc_output, enc_hidden = model.call_encoder(inp) # is this necessary? !!!
        # call(self, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None)
        # final_dist [batch_sz, max_len_y, extend_vocab_size] attentions, coverages [max_len_y, batch_sz, max_len_x, 1]
        final_dist, attentions, coverages = model(inp, targ, enc_extended_inp, batch_oov_len, use_coverage=True, prev_coverage=None)
        loss = loss_function(targ, final_dist, padding_mask) + cov_loss_wt * coverage_loss(attentions, coverages, padding_mask)
        batch_loss = (loss / int(targ.shape[1]))
    variables = model.trainable_variables
    # print(variables)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss




w2v_model = Word2Vec.load('./word2vec.model')
max_lens = [243, 244, 33, 254]
#embedding_matrix = get_embedding_matrix(w2v_model)
embedding_matrix = np.loadtxt('embedding_matrix.txt', dtype=np.float32)
batch_sz = 64

train_X = []
train_X_oov = []
train_y = []
train_y_oov = []
with open('./data/train_X_pad.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_X.append(line.strip().split(' '))
f.close()
with open('./data/train_X_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_X_oov.append(line.strip().split(' '))
f.close()
with open('./data/train_y_pad.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_y.append(line.strip().split(' '))
f.close()
with open('./data/train_y_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        train_y_oov.append(line.strip().split(' '))
f.close()

train_X_token, train_X_extended_token, train_X_pad_mask, train_X_oov_dict, train_X_oov_len = get_token(w2v_model, max_lens[1], train_X, train_X_oov)
train_y_token, _, train_y_pad_mask, _, _ = get_token(w2v_model, max_lens[2], train_y)

dataset_train_batch, dataset_test_batch = batch(64, train_X_token, train_X_extended_token, train_X_pad_mask, train_X_oov_len, train_y_token, train_y_pad_mask)
example_input_batch, example_enc_extend, example_enc_mask,example_oov_len, example_target_batch, example_target_mask = next(iter(dataset_train_batch))

gru_units = 256
att_units = 50
embedding_matrix = embedding_matrix

pgn_model = PGN(gru_units, att_units, batch_sz, embedding_matrix)
# final_dist, attentions, coverages = pgn_model(example_input_batch, example_target_batch, example_enc_extend, example_oov_len, use_coverage=True, prev_coverage=None)
print('finish')

optimizer = tf.keras.optimizers.Adam(clipvalue=2.0)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# loss_one_step = train_one_step(pgn_model, example_input_batch, example_target_batch, example_enc_extend, example_oov_len, cov_loss_wt = 1.0, padding_mask=example_target_mask)
# print(loss_one_step)

EPOCHS = 5
steps_per_epoch = 70761 // batch_sz #!!!
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint()

for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    for (batch, (example_input_batch, example_enc_extend, example_enc_mask,example_oov_len, example_target_batch,
                 example_target_mask)) in enumerate(dataset_test_batch.take(steps_per_epoch)):
        batch_loss = train_one_step(pgn_model, example_input_batch, example_target_batch, example_enc_extend,
                                    example_oov_len, cov_loss_wt = 1.0, padding_mask=example_target_mask)
        total_loss += batch_loss

        if batch % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
>>>>>>> a084f9ce23303f1bf8968fa732051cb08c426656
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
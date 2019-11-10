# summerization_with_seq2seq_implementing_pgn_and_coverage
## 1. data_loader.py do purge data, segment sentence, add special token, train word2vec model and finally get embedding. I also added a tf-idf filter, though its effect is waiting for experiment. I do not upload the embedding matrix file because it is >100 MB after I change the vocab size and embedding dim, you need to generate it by yourself.
## 2. model_layers.py and pgn.py define seq2seq model structure implementing coverage and p_gen almost without option.
## 3. train.py token input, make batch and train the model.
## 4. Note this is a draft version and I am trying debug and new settings. The parameters may hide in somewhere so be very careful. :)
## 5. Good luck. :)

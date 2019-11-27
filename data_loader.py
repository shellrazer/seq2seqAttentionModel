# -*- coding: utf-8 -*-

import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

pd.set_option('display.max_columns', 1000)
pd.set_option("display.max_colwidth", 1000)

def data_generate(paths):
    train_path = paths['train_path']
    test_path = paths['test_path']
    train_text_path = paths['train_text_path']
    test_text_path = paths['test_text_path']
    train_X_path = paths['train_X_path']
    train_y_path = paths['train_y_path']
    test_X_path = paths['test_X_path']
    corpus_path = paths['train_test_merged_path']

    train_set = pd.read_csv(train_path, dtype=str, encoding = 'utf-8')
    test_set = pd.read_csv(test_path, dtype=str, encoding = 'utf-8')

    # remove unnecessary word
    for series in ['Question', 'Dialogue', 'Report']:
        train_set[series] = train_set[series].str.replace(
            "[a-zａ-ｚA-ZＡ-Ｚ0-9０-９]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好", '',
            regex=True)
    train_set.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    # combine for word2vec corpus
    train_set['Comb'] = train_set[['Question', 'Dialogue']].apply(lambda x: ''.join(x), axis=1)
    train_set['Comb'].to_csv(train_text_path, index=False, header=False, encoding='utf-8')
    # combine for train dataset
    train_set = train_set.loc[train_set.Report.str.len() > 5, :]
    train_set['X'] = train_set[['Question', 'Dialogue']].apply(lambda x: ''.join(x), axis=1)
    train_set['X'].to_csv(train_X_path, index=False, header=False, encoding='utf-8')
    train_set['Report'].to_csv(train_y_path, index=False, header=False, encoding='utf-8')
    print("train_data_generation done!")

    # remove unnecessary word
    for series in ['Question', 'Dialogue']:
        test_set[series] = test_set[series].str.replace(
            "[a-zａ-ｚA-ZＡ-Ｚ0-9０-９]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好", '',
            regex=True)
    test_set.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    # combine for word2vec corpus
    test_set['X'] = test_set[['Question', 'Dialogue']].apply(lambda x: ''.join(x), axis=1)
    test_set['X'].to_csv(test_text_path, index=False, header=False, encoding='utf-8')
    test_set['X'].to_csv(test_X_path, index=False, header=False, encoding='utf-8')
    print("test_data_generation done!")

    corpus_text = []
    with open(train_text_path,encoding='utf-8',errors='ignore') as f:
        line_list = f.readlines()
        for line in line_list:
            corpus_text.append(line.strip())
        f.close()
    with open(test_text_path,encoding='utf-8',errors='ignore') as f:
        line_list = f.readlines()
        for line in line_list:
            corpus_text.append(line.strip())
        f.close()
    with open(corpus_path, 'w', encoding = 'utf-8') as f:
        for line in corpus_text:
            f.write(line)
            f.write('\n')
    print('file saved')


def get_segment(paths):

    data_in_paths = paths['to_segment'] #list
    data_out_paths = paths['after_segment'] #list
    userdict_path = paths['userdict_path']
    stop_words_path = paths['stop_words']

    # stopwords = []
    # with open(stop_words_path, encoding='utf-8', errors='ignore') as f:
    #     for line in f.readlines():
    #         stopwords.append(line.strip())
    #     stopword_set = set(stopwords)
    #     print('停顿词列表，stopwords中共有%d个元素' % len(stopwords))
    #     print('停顿词集合，stopword_set中共有%d个元素' % len(stopword_set))
    #     f.close()
    stopword_set = ['的','了','有']
    max_lens = []
    # data_segment_list holds segmented dataset data_segment_length holds length of each sample in dataset
    # max_lens hold the max input or output length of dataset, the order is consistent with paths
    jieba.load_userdict(userdict_path)
    for index, data_in_path in enumerate(data_in_paths):
        print('start processing {}'.format(data_in_path))
        data_segment_list = []
        data_segment_length = []
        with open(data_in_path, encoding='utf-8', errors='ignore') as f:
            line_list = f.readlines()
            data_list = [line.strip() for line in line_list]
            f.close()
        for text in data_list:
            text = str(text)
            cut_words = [word for word in jieba.cut(text) if word not in stopword_set]
            data_segment_list.append(cut_words)
            data_segment_length.append(len(cut_words))
        max_len = int(np.mean(data_segment_length) + 2*np.std(data_segment_length))
        max_lens.append(max_len)
        print('segment are belong to {path} has {len} samples, mean length of samples is {mean} and choice {max_len} '
              'as length of input'.format(path=data_in_path, len=len(data_segment_list),
                                          mean=int(np.mean(data_segment_length)), max_len=max_len))
        save_files(data_segment_list, data_out_paths[index])

    return max_lens


def save_files(list, path):
    with open(path, 'w', encoding = 'utf-8') as f:
        for line in list:
            line = ' '.join(line)
            f.write(line)
            f.write('\n')
    print('file saved')


def tfidf_filter(in_path, max_df, min_df, min_tfidf):
    dataset = []
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dataset.append(line.strip())
    f.close()
    print('dataset sample number:',len(dataset))
    # print(dataset[9].split(' '))
    # print(len(dataset[9].split(' ')))
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=max_df, min_df=min_df)
    tfidf_matrix = tfidf.fit_transform(dataset)
    #print(tfidf.vocabulary_)
    print('original vocabulary size:',len(tfidf.vocabulary_))
    # print(len(tfidf.get_feature_names()))
    # print('顺时针' in tfidf.vocabulary_)
    # print(tfidf.vocabulary_['顺时针'])
    #print(tfidf_matrix[1000:1500].toarray())
    new_dataset = []
    line_length = []
    #count = 0
    for index, line in enumerate(dataset):
        new_line=[]
        for word in line.split(' '):
            if word in tfidf.vocabulary_ and tfidf_matrix[index].toarray()[0][tfidf.vocabulary_[word]] > min_tfidf:
                new_line.append(word)
        #count+=1
        #print(count,'lines are processed')
        new_dataset.append(new_line)
        line_length.append(len(new_line))
    max_len = int(np.mean(line_length) + 2*np.std(line_length))
    print('max_len', max_len)
    print('new dataset sample number',len(new_dataset))
    # print(new_dataset[0:10])
    with open(in_path, 'w', encoding='utf-8') as f:
        for line in new_dataset:
            line = ' '.join(line)
            f.write(line)
            f.write('\n')
    print('new file saved')
    return max_len

#################END OF DATA PURGE & START OF VOCAB AND TONKENIZE#################

def prepare_dataset(paths, embedding_size,max_lens):
    #_, max_train_inp, max_train_out, max_test_inp = max_lens
    # train the word2vec model and obtain vocab without start end pad unk token
    merge, train_X, train_y, test_X = paths['after_segment']
    #_,train_X_oov, train_y_oov, test_X_oov = paths['dataset_oovs']
    print('start build w2v model')
    w2v_model = Word2Vec(LineSentence(merge), size=embedding_size, negative=5, workers=4, iter=100, window=3, min_count=1)
    #w2v_model.save('./word2vec.model')
    #w2v_model = Word2Vec.load('./word2vec.model')
    print('finish build w2v model')
    print('w2v_model has vocabulary of ', len(w2v_model.wv.vocab))
    # now we add <start> <end> <pad> <unk> token, prepare sample with right length and retrain word2vec
    for i in range(1, len(max_lens)):
        path = paths['after_segment'][i]
        pad_path = paths['after_pad'][i]
        oov_path = paths['dataset_oovs'][i]
        max_len = max_lens[i] + 2 # plus <START> <STOP>
        newlines = []
        dataset_oov = []
        with open(path, 'r', encoding='utf-8') as f:
            for k in f.readlines():
                in_article_oov = []
                new_word_list = ['<START>']
                word_list = k.strip().split(' ')
                if max_len - 2 >= len(word_list):
                    for word in word_list:
                        if word in w2v_model.wv.vocab:
                            new_word_list.append(word)
                        else:
                            new_word_list.append('<UNK>')
                            in_article_oov.append(word)
                    new_word_list.append('<STOP>')
                    for _ in range(max_len - 2 - len(word_list)):
                        new_word_list.append('<PAD>')
                else:
                    for index in range(max_len - 2):
                        if word_list[index] in w2v_model.wv.vocab:
                            new_word_list.append(word_list[index])
                        else:
                            new_word_list.append('<UNK>')
                            in_article_oov.append(word_list[index])
                    new_word_list.append('<STOP>')
                newline = ' '.join(new_word_list)
                newlines.append(newline)
                dataset_oov.append(in_article_oov)
                assert len(new_word_list) == max_len
            assert len(dataset_oov) == len(newlines)

        with open(pad_path, 'w', encoding='utf-8') as f:
            for line in newlines:
                f.write(line)
                f.write('\n')
        with open(oov_path, 'w', encoding='utf-8') as f:
            for oov in dataset_oov:
                f.write(' '.join(oov))
                f.write('\n')
        f.close()
        max_lens[i] = max_len

    print('start retrain w2v model')
    w2v_model.build_vocab(LineSentence(paths['after_pad'][3]), update=True)
    w2v_model.train(LineSentence(paths['after_pad'][3]), epochs=50, total_examples=w2v_model.corpus_count)
    print('1/3')
    w2v_model.build_vocab(LineSentence(paths['after_pad'][1]), update=True)
    w2v_model.train(LineSentence(paths['after_pad'][1]), epochs=50, total_examples=w2v_model.corpus_count)
    print('2/3')
    w2v_model.build_vocab(LineSentence(paths['after_pad'][2]), update=True)
    w2v_model.train(LineSentence(paths['after_pad'][2]), epochs=50, total_examples=w2v_model.corpus_count)
    w2v_model.save('./word2vec.model')
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(w2v_model.wv.vocab))
    return w2v_model, max_lens

def get_token(w2v_model, max_len_x, dataset, oovs=None):
    """
    dataset, list of lists, number of samples*max_len_x; oovs, list of list, number of samples * len of oov(not fixed)
    return: tokenized dataset, tonkenized extended dataset(where unk is represented by extended index), dataset pad mask
    """

    vocab_size = len(w2v_model.wv.vocab)

    START_index = w2v_model.wv.vocab['<START>'].index
    STOP_index = w2v_model.wv.vocab['<STOP>'].index
    PAD_index = w2v_model.wv.vocab['<PAD>'].index
    UNK_index = w2v_model.wv.vocab['<UNK>'].index
    Special_word = ('<START>','<STOP>','<PAD>','<UNK>')
    dataset_token = []
    dataset_extended_token = []
    dataset_pad_mask = []
    if oovs is not None:
        # [number of samples, len of unique oov words(not fix)]
        dataset_oov_dict = [list(set(oov)) for oov in oovs]
        # [number of samples, 1]
        dataset_oov_len = [len(oov) if oov != [''] else 0 for oov in dataset_oov_dict]
    else:
        dataset_oov_dict = []
        dataset_oov_len =[]


    for sample_index, sample in enumerate(dataset):
        oov_count = 0
        sample_token = []
        sample_extended_token = []
        sample_pad_mask = [1 for _ in range(max_len_x)]
        for word_index, word in enumerate(sample):
            if word == '<UNK>':
                sample_token.append(UNK_index)
                if oovs is not None:
                    oov_id = dataset_oov_dict[sample_index].index(oovs[sample_index][oov_count])
                    sample_extended_token.append(vocab_size + oov_id)
                    oov_count += 1   # oov_count count the #rd oov word
            elif word == '<PAD>':
                sample_token.append(PAD_index)
                sample_extended_token.append(PAD_index)
                sample_pad_mask[word_index] = 0
            else:
                sample_token.append(w2v_model.wv.vocab[word].index)
                sample_extended_token.append(w2v_model.wv.vocab[word].index)
        dataset_token.append(sample_token)
        dataset_extended_token.append(sample_extended_token)
        dataset_pad_mask.append(sample_pad_mask)
    return dataset_token, dataset_extended_token, dataset_pad_mask, dataset_oov_dict, dataset_oov_len

def token_to_word(w2v_model, tokens, oov_dict):

    # oov is a word list oov_dict

    START_index = w2v_model.wv.vocab['<START>'].index
    STOP_index = w2v_model.wv.vocab['<STOP>'].index
    PAD_index = w2v_model.wv.vocab['<PAD>'].index
    UNK_index = w2v_model.wv.vocab['<UNK>'].index
    vocab_size = len(w2v_model.wv.vocab)
    word = ''
    # if type(tokens) == int:
    #     if tokens < vocab_size:
    #         word = word + w2v_model.wv.index2word[tokens]
    #     elif tokens < vocab_size + len(oov_dict):
    #         word = word + oov_dict[tokens - vocab_size]
    #     else:
    #         word = word + "<UNK>"
    # else:
    for token in tokens:
        if token < vocab_size:
            word = word + w2v_model.wv.index2word[token]
        elif token < vocab_size + len(oov_dict):
            word = word + oov_dict[token - vocab_size]
        else:
            word = word + "<UNK>"
        if token == STOP_index:
            break

    return word

def get_embedding_matrix(w2v_model):
    vocab_size = len(w2v_model.wv.vocab)
    embedding_dim = len(w2v_model.wv['<START>'])
    print('vocab_size, embedding_dim:', vocab_size, embedding_dim)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    print('start extract embedding matrix, may take long time')
    for i in range(vocab_size):
        embedding_matrix[i, :] = w2v_model.wv[w2v_model.wv.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    assert embedding_matrix.shape == (vocab_size, embedding_dim)
    np.savetxt('embedding_matrix.txt', embedding_matrix, fmt='%0.8f')
    print('embedding matrix extracted')
    return embedding_matrix


def batch(BATCH_SIZE, test_size, input, extended_input, input_pad_mask, input_oov_dict, input_oov_len, output, output_pad_mask):


    input_train, input_test, extended_input_train, extended_input_test, input_pad_mask_train, input_pad_mask_test, \
    input_oov_train_dict, input_oov_test_dict, input_oov_train_len, input_oov_test_len, output_train, output_test, \
    output_pad_mask_train, output_pad_mask_test = train_test_split(input, extended_input, input_pad_mask,
                                                                   input_oov_dict, input_oov_len,output,output_pad_mask,
                                                                   test_size=test_size, random_state=6)

    train_dataset_len = len(input_train)
    test_dataset_len = len(input_test)

    input_train, input_test, extended_input_train, extended_input_test, input_pad_mask_train, input_pad_mask_test, \
    input_oov_train_len, input_oov_test_len, output_train, output_test, output_pad_mask_train, output_pad_mask_test = \
    tf.convert_to_tensor(input_train), tf.convert_to_tensor(input_test), \
    tf.convert_to_tensor(extended_input_train), tf.convert_to_tensor(extended_input_test), \
    tf.convert_to_tensor(input_pad_mask_train), tf.convert_to_tensor(input_pad_mask_test), \
    tf.convert_to_tensor(input_oov_train_len), tf.convert_to_tensor(input_oov_test_len), \
    tf.convert_to_tensor(output_train), tf.convert_to_tensor(output_test), \
    tf.convert_to_tensor(output_pad_mask_train), tf.convert_to_tensor(output_pad_mask_test)

    print('train_test_split, train_input shape:', input_train.shape)
    print('train_test_split, test_input shape:', input_test.shape)

    dataset_train = tf.data.Dataset.from_tensor_slices(
    (input_train, extended_input_train, input_pad_mask_train, input_oov_train_len, output_train, output_pad_mask_train))
    dataset_test = tf.data.Dataset.from_tensor_slices(
    (input_test, extended_input_test, input_pad_mask_test, input_oov_test_len, output_test, output_pad_mask_test))
    dataset_train_batch = dataset_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    dataset_test_batch = dataset_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    return dataset_train_batch, dataset_test_batch, train_dataset_len, test_dataset_len, input_oov_train_dict, input_oov_test_dict


def pip_data(params):
    data_dir = params['data_dir']
    max_df = params['max_df']
    min_df = params['min_df']
    min_tfidf = params['min_tfidf']
    embedding_size = params['embedding_size']

    paths = {
    'train_path': os.path.join(data_dir, 'AutoMaster_TrainSet.csv'),
    'test_path': os.path.join(data_dir, 'AutoMaster_TestSet.csv'),
    'train_text_path': os.path.join(data_dir, 'train_text.txt'),
    'test_text_path' : os.path.join(data_dir, 'test_text.txt'),
    'train_X_path' : os.path.join(data_dir, 'train_X.txt'),
    'train_y_path' : os.path.join(data_dir, 'train_y.txt'),
    'test_X_path' : os.path.join(data_dir, 'test_X.txt'),
    'stop_words' : os.path.join(data_dir, 'stop_words.txt'),
    'train_test_merged_path':os.path.join(data_dir, 'merged_train_test.txt'),
    'userdict_path':os.path.join(data_dir, 'user_dict.txt'),
    'to_segment': [os.path.join(data_dir, 'merged_train_test.txt'),
                   os.path.join(data_dir, 'train_X.txt'),
                   os.path.join(data_dir, 'train_y.txt'),
                   os.path.join(data_dir, 'test_X.txt')],
    'after_segment': [os.path.join(data_dir, 'merged_train_test_segment.txt'),
                   os.path.join(data_dir, 'train_X_segment.txt'),
                   os.path.join(data_dir, 'train_y_segment.txt'),
                   os.path.join(data_dir, 'test_X_segment.txt')],
    'dataset_oovs': [os.path.join(data_dir, 'merged_train_test_oov.txt'),
                   os.path.join(data_dir, 'train_X_oov.txt'),
                   os.path.join(data_dir, 'train_y_oov.txt'),
                   os.path.join(data_dir, 'test_X_oov.txt')],
    'after_pad': [os.path.join(data_dir, 'merged_train_test_pad.txt'),
                   os.path.join(data_dir, 'train_X_pad.txt'),
                   os.path.join(data_dir, 'train_y_pad.txt'),
                   os.path.join(data_dir, 'test_X_pad.txt')],
}

    data_generate(paths)
    max_lens = get_segment(paths)
    print(max_lens)
    # max_lens = [339, 337, 39, 355]
    # I am not sure max_lens really works. It do bypassed the oov problem.
    # tfidf_filter naively iterate every word in corpus, is extremely slow! !!!!!danger!!!!!! ~ 2 hour waiting
    # tfidf_filter overwrite the segment file, people can bypass this step and use segment file directly.
    for index, file_path in enumerate(paths['after_segment']):
        max_lens[index] = tfidf_filter(file_path, max_df, min_df, min_tfidf)
        #vocab merge 54225 -> 47792  train_X 47488 -> 41661  train_y 12279 -> 12267 test_X 22356 -> 18673
    print(max_lens)
    # max_lens = [98, 98, 32, 101]
    w2v_model, max_lens = prepare_dataset(paths, embedding_size, max_lens)  # vocab_sz 53465 -> 53469
    print('max lens:', max_lens)  #  [98, 100, 34, 103]
    w2v_model = Word2Vec.load('./word2vec.model')
    #max_lens = [241, 87, 29, 87]

    embedding_matrix = get_embedding_matrix(w2v_model) #vocab_size 53469 embedding_dim 256
    #embedding_matrix = np.loadtxt('embedding_matrix.txt',dtype=np.float32)
    #print(np.sum(embedding_matrix - embedding_matrix_2))


#     train_X = []
#     train_X_oov = []
#     train_y = []
#     train_y_oov = []
#     with open('./data/train_X_pad.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             train_X.append(line.strip().split(' '))
#     f.close()
#     with open('./data/train_X_oov.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             train_X_oov.append(line.strip().split(' '))
#     f.close()
#     with open('./data/train_y_pad.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             train_y.append(line.strip().split(' '))
#     f.close()
#     with open('./data/train_y_oov.txt', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             train_y_oov.append(line.strip().split(' '))
#     f.close()
#
#     train_X_token, train_X_extended_token, train_X_pad_mask, train_X_oov_dict, train_X_oov_len = get_token(
#         w2v_model, max_lens[1], train_X, train_X_oov)
#     train_y_token, _, train_y_pad_mask, _, _ = get_token(w2v_model, max_lens[2], train_y)
#
#     print(len(train_X_token),len(train_X_extended_token),len(train_X_pad_mask),len(train_X_oov_dict),len(train_X_oov_len),len(train_y_token),len(train_y_pad_mask))
#     print(train_X[35])
#     print(train_X_oov[35])
#     print(train_y[35])
#     print(len(train_X[35]), len(train_X_oov[35]), len(train_y[35]), len(train_y_oov[35]))
#     print(train_X_token[35])
#     print(train_X_extended_token[35])
#     print(train_y_token[35])
#     print(len(train_X_token[35]), len(train_X_extended_token[35]), len(train_y_token[35]))
#     print(token_to_word(w2v_model, train_X_token[35], train_X_oov_dict[35]))
#     print(token_to_word(w2v_model, train_X_extended_token[35], train_X_oov_dict[35]))
#     print(len(token_to_word(w2v_model, train_X_token[35], train_X_oov_dict[35])))
#     print(len(token_to_word(w2v_model, train_X_extended_token[35], train_X_oov_dict[35])))
#     print(token_to_word(w2v_model, train_y_token[35], train_X_oov_dict[35]))
#     print(len(token_to_word(w2v_model, train_y_token[35], train_X_oov_dict[35])))
#
#
#     dataset_train_batch, dataset_test_batch = batch(
#         64, train_X_token, train_X_extended_token,train_X_pad_mask, train_X_oov_len, train_y_token, train_y_pad_mask)
#     example_input_batch, example_enc_extend, example_enc_mask,example_oov_len, example_target_batch, \
#     example_target_mask = next(iter(dataset_train_batch))
#     print (example_input_batch.shape, example_enc_extend.shape, example_oov_len.shape, example_target_batch.shape)
#
#
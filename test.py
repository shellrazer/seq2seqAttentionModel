# # import gensim.downloader as api
# # from gensim.models import TfidfModel
# # from gensim.corpora import Dictionary
# #
# #
# # #dataset = api.load('./data/merged_train_test_segment.txt')
# # dataset = []
# # with open('./data/merged_train_test_segment.txt', 'r', encoding='utf-8') as f:
# #     for line in f.readlines():
# #         dataset.append(line.strip().split(' '))
# # f.close()
# # print('69th line:', dataset[69])
# # dct = Dictionary(dataset)
# # len(dct)
# # corpus = [dct.doc2bow(line) for line in dataset]
# # print('corpus 69th line',corpus[69])
# # tfidf = TfidfModel(corpus, id2word=dct)
# # low_value = 0.02
# # low_value_words = []
# # for bow in corpus:
# #     low_value_words += [id for id, value in tfidf[bow] if value < low_value]
# # #print('low value words:',low_value_words)
# # print('its length:',len(low_value_words))
# # dct.filter_tokens(bad_ids=low_value_words)
# # new_corpus = [dct.doc2bow(line) for line in dataset]
# # print('new_corpus 69th line',new_corpus[69])
# # # with open('./filtered_text', 'w', encoding='utf-8') as f:
# # #     for line in new_corpus:
# # #         f.write(line)
# # #         f.write('\n')
# # # f.close()
#
#
# # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# #
# # dataset = []
# # with open('./data/test_X_segment.txt', 'r', encoding='utf-8') as f:
# #     for line in f.readlines():
# #         dataset.append(line.strip())
# # f.close()
# # print(len(dataset))
# # print(dataset[9].split(' '))
# # print(len(dataset[9].split(' ')))
# # tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.75,min_df=2)
# # tfidf_matrix = tfidf.fit_transform(dataset)
# # print(tfidf.vocabulary_)
# # print(len(tfidf.vocabulary_))
# # print(len(tfidf.get_feature_names()))
# # print('顺时针' in tfidf.vocabulary_)
# # print(tfidf.vocabulary_['顺时针'])
# # print(tfidf_matrix[9].toarray()[0][53064])
# # new_dataset = []
# # count = 0
# # for index, line in enumerate(dataset):
# #     new_line=[]
# #     for word in line.split(' '):
# #         if word in tfidf.vocabulary_ and tfidf_matrix[index].toarray()[0][tfidf.vocabulary_[word]] > 0.02:
# #             new_line.append(word)
# #     count+=1
# #     print(count,'lines are processed')
# #     new_dataset.append(new_line)
# # print(len(new_dataset))
# # print(new_dataset[9])
# # with open('./data/new_segment', 'w', encoding = 'utf-8') as f:
# #     for line in new_dataset:
# #         line = ' '.join(line)
# #         f.write(line)
# #         f.write('\n')
# # print('file saved')
# # from gensim.models import Word2Vec
# # from gensim.models.word2vec import LineSentence
# # merge = './data/merge_1.txt'
# # merge_2 = './data/merge_2.txt'
# # w2v_model = Word2Vec(LineSentence(merge), size=10, negative=5, workers=4, iter=10, min_count=1)
# # print(len(w2v_model.wv.vocab))
# # print(w2v_model.wv['没有'])
# # print(w2v_model.wv['价格'])
# # print(w2v_model.wv['希望'])
# # w2v_model.build_vocab(LineSentence(merge_2), update=True)
# # w2v_model.train(LineSentence(merge_2), epochs=30, total_examples=w2v_model.corpus_count)
# # print(w2v_model.wv['没有'])
# # print(w2v_model.wv['价格'])
# # print(w2v_model.wv['希望'])
# # print(w2v_model.wv['钣金'])
# # print(len(w2v_model.wv.vocab))
#
# # merge = './data/merge_1.txt'
# # a=[]
# # with open(merge, 'r', encoding='utf-8') as f:
# #     for line in f.readlines():
# #         a.append(line.strip())
# # f.close()
# # print(a)
# #
# # new_dataset= [['a','b','c'],['1','2','4',' ', 'a']]
# #
# # with open(merge, 'w', encoding='utf-8') as f:
# #     for line in new_dataset:
# #         line = ' '.join(line)
# #         f.write(line)
# #         f.write('\n')
#
#
# from data_loader import get_token, token_to_word
# from gensim.models import Word2Vec
#
# train_X = []
# train_X_oov = []
# train_X_path = './data/train_y_pad.txt'
# train_X_oov_path = './data/train_y_oov.txt'
# w2v_model = Word2Vec.load('./word2vec.model')
#
# print(len(w2v_model.wv.vocab))
# with open(train_X_path, 'r', encoding='utf-8') as f:
#     for line in f.readlines()[0:10]:
#         train_X.append(line.strip().split(' '))
# f.close()
# with open(train_X_oov_path, 'r', encoding='utf-8') as f:
#     for line in f.readlines()[0:10]:
#         train_X_oov.append(line.strip().split(' '))
# f.close()
# print(train_X_oov[1])
# print(train_X_oov[1] is [])
# print(train_X_oov[1] is '')
# print(train_X_oov[1] is [''])
# print(train_X_oov[1] is list(''))
# print(train_X_oov[1] == 0)
# print(train_X_oov[1] == [])
# print(train_X_oov[1] == [''])
# print(train_X_oov[1] == '')
#
# dataset_token, dataset_extended_token, dataset_pad_mask, dataset_oov_dict, dataset_oov_len = get_token(w2v_model, 37, train_X, train_X_oov)
# for i in range(len(dataset_token)):
#     print(token_to_word(w2v_model, dataset_token[i], dataset_oov_dict[i]))
#     print(token_to_word(w2v_model, dataset_extended_token[i], dataset_oov_dict[i]))
# #print(dataset_token, dataset_extended_token, dataset_pad_mask, dataset_oov_dict, dataset_oov_len)
# print('finish')

import argparse

parser = argparse.ArgumentParser()

# By default it will fail with multiple arguments.
parser.add_argument('--default')

# Telling the type to be a list will also fail for multiple arguments,
# but give incorrect results for a single argument.
parser.add_argument('--list-type', default=[1,2,3,4], type=list)

# This will allow you to provide multiple arguments, but you will get
# a list of lists which is not desired.
parser.add_argument('--list-type-nargs', type=list, nargs='+')

# This is the correct way to handle accepting multiple arguments.
# '+' == 1 or more.
# '*' == 0 or more.
# '?' == 0 or 1.
# An int is an explicit number of arguments to accept.
parser.add_argument('--nargs', nargs='+')

# To make the input integers
parser.add_argument('--nargs-int-type', default=[1,1,1,1], nargs='+', type=int)

# An alternate way to accept multiple inputs, but you must
# provide the flag once per input. Of course, you can use
# type=int here if you want.
parser.add_argument('--append-action', action='append')

# To show the results of the given option to screen.
for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        print(value)

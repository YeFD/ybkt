# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:22
# @File    : score_w2v-SGD.py
'''
求word2vec+SGD的默认ACC
'''
import jieba
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

stopwords_path = 'stopword.txt'
pos_path = 'pos.txt'
neg_path = 'neg.txt'
w2v_model = joblib.load('w2v.model')

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg  # sentence list
# label = [1]*16548 + [0]*18581
# print(stopwords)


def del_stopwords(sentence):
    result = []
    for word in sentence:
        if word in stopwords:
            continue
        else:
            result.append(word)
    return result


def cut_sentence(sentence):
    word_list = jieba.cut(sentence)
    word_list = del_stopwords(word_list)
    return word_list


def get_w2v(word_list):  # 传入一个由n个单词组成的word_list，返回一个由n个向量组成的list，每个向量对应一个单词
    w2v_list = []
    for word in word_list:
        word = word.replace('\n', '')  # 删除换行符
        try:
            w2v_list.append(w2v_model[word])
        except KeyError:
            continue
    return np.array(w2v_list, dtype='float')


def build_w2v(sentence_list):  # 传入一个由n个句子组成的sentence_list，返回由n个array向量组成的list，每个array向量对应一个句子
    sentence_w2v_list = []
    for sentence in sentence_list:
        word_list = cut_sentence(sentence)  # 句子分词,返回一个由n个单词组成的word_list
        word_w2v_list = get_w2v(word_list)  # 返回单词向量list
        if len(word_w2v_list) != 0:
            w2v_array = sum(np.array(word_w2v_list))/len(word_w2v_list)  # 单词向量相加，将w2v_list合并为为一个句子array向量
            sentence_w2v_list.append(w2v_array)
    return sentence_w2v_list


pos_w2v = build_w2v(pos)  # 16540 list[array([])]
neg_w2v = build_w2v(neg)  # 18561
data_all = build_w2v(data)  # 35101
data_X = np.array(data_all)
data_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))

train_data, test_data, train_label, test_label = train_test_split(data_X, data_label, test_size=0.3, random_state=2)

SGD_model = SGDClassifier(loss='log', penalty='l1', alpha=0.01)
# loss='log'-逻辑回归  'hinge'-svm
# penalty='l1'  'l2'  'penalty'  'none'
# alpha default=0.0001
# n_iter default=5
SGD_model.fit(train_data, train_label)

acc = SGD_model.score(test_data, test_label)
print(acc)
# loss='log', penalty='l1'         0.7289905991833634
# loss='log', penalty='l2'         0.7283258949767354
# loss='log', penalty='elasticnet' 0.7128477827366821
# loss='hinge', penalty='l1'       0.7126578672490742
# loss='hinge', penalty='l2'       0.7275662330263033
# loss='log', penalty='l1', alpha=0.001  0.7351628525306239

# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:23
# @File    : score_w2v-bayes.py
'''
求word2vec+GaussianNB的默认ACC
'''
import jieba
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

pos_path = 'pos.txt'
neg_path = 'neg.txt'
stopwords_path = 'stopword.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg  # sentence list
w2v_model = joblib.load('w2v.model')


def sigmoid(x):
    return 1/(1+np.exp(-x))


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


def build_w2v(sentence_list):
    result = []
    for sentence in sentence_list:
        word_list = cut_sentence(sentence)
        w2v_list = get_w2v(word_list)
        if len(w2v_list) != 0:
            result_list = sum(np.array(w2v_list))/len(w2v_list)  # GNB
            # result_list = sigmoid(sum(w2v_list)/len(w2v_list))  # 单位化
            result.append(result_list)
    return result


pos_w2v = build_w2v(pos)  # 16540
neg_w2v = build_w2v(neg)  # 18561
# data_train = pos_w2v + neg_w2v  # 35101
data_all = build_w2v(data)  # 35101

print(data_all)
label = [1]*len(pos_w2v) + [0]*len(neg_w2v)
train_data, test_data, train_label, test_label = train_test_split(data_all, label, test_size=0.3, random_state=2)

model = GaussianNB()
model.fit(train_data, train_label)
acc = model.score(test_data, test_label)
print(acc)  # 0.6996486563479252 单位化后0.696704966290001

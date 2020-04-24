# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:25
# @File    : validation_curve_w2v-svm.py
'''
绘制SVM的学习曲线
'''
import jieba
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
pos_path = 'pos.txt'
neg_path = 'neg.txt'
stopwords_path = 'stopword.txt'
w2v_model = joblib.load('w2v.model')
start = time.process_time()

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg
# print(stopwords)
# print(len(neg_train))


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


def get_w2v(word_list):
    w2v_list = []
    for word in word_list:
        word = word.replace('\n', '')
        try:
            w2v_list.append(w2v_model[word])
        except KeyError:
            continue
    return np.array(w2v_list, dtype='float')


def build_w2v(sentence_list):
    sentence_w2v_list = []
    for sentence in sentence_list:
        word_list = cut_sentence(sentence)
        word_w2v_list = get_w2v(word_list)
        if len(word_w2v_list) != 0:
            w2v_array = sum(np.array(word_w2v_list))/len(word_w2v_list)
            sentence_w2v_list.append(w2v_array)
    return sentence_w2v_list


print(time.process_time() - start)
pos_w2v = build_w2v(pos)
neg_w2v = build_w2v(neg)
train_data = build_w2v(data)
print(time.process_time() - start)
train_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))
train_X = np.array(train_data)
print(time.process_time() - start)

svm_model = SVC(kernel='rbf', probability=False, gamma='auto')
# svm_model.fit(train_X, train_label)
print(time.process_time() - start)

param_range = np.logspace(-2, 2, 25)
# param_range = np.arange(1, 20, 20)
train_scores, test_scores = validation_curve(svm_model, train_X, train_label, param_name='C', param_range=param_range, cv=10, scoring='accuracy', n_jobs=-1)
print(time.process_time() - start)

train_scores_mean = np.mean(train_scores, axis=1)  # axis=1, 就算每一行的均值
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$C$')
plt.ylabel('Score')
plt.ylim(0.5, 1.0)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Traing score',
              color='darkorange', lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color='darkorange', lw=lw)
plt.semilogx(param_range, test_scores_mean, label='Cross-calidation score',
              color='navy', lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color='navy', lw=lw)

plt.legend(loc='best')
plt.show()

print(time.process_time() - start)

# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:25
# @File    : validation_curve_TFIDF-bayes.py
'''
绘制MultinomialNB的验证曲线
'''
import jieba
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_path = 'stopword.txt'
pos_path = 'pos.txt'
neg_path = 'neg.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
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


def cut_words(sentence_list):
    result = []
    for sentence in sentence_list:
        word_list = jieba.cut(sentence)  # 分词-精确切分 cut_all=False
        word_list = del_stopwords(word_list)
        sentence_cut = ' '.join(word_list)
        result.append(sentence_cut)
    return result


TFIDF_model = TfidfVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}',
                              ngram_range=(1, 4),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1,
                              stop_words=None)

pos_cut = cut_words(pos)
neg_cut = cut_words(neg)
label = [1] * 16548 + [0] * 18581
data_all = pos_cut + neg_cut  # 16548 18581
len_pos = len(pos_cut)
TFIDF_model.fit(data_all)
data_train = TFIDF_model.transform(data_all)

model = MultinomialNB()
param_range = np.logspace(-7, 2, 20)
# param_range = np.arange(0.01, 2, 0.05)
train_scores, test_scores = validation_curve(model, data_train, label, param_name='alpha', param_range=param_range, cv=10, scoring='accuracy', n_jobs=10)

train_scores_mean = np.mean(train_scores, axis=1)  # axis=1, 就算每一行的均值
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with MultinomialNB')
plt.xlabel('$alpha$')
plt.ylabel('Score')
plt.ylim(0.825, 1.0)
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
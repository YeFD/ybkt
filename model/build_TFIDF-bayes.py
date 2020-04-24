# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:19
# @File    : build_TFIDF-bayes.py
'''
需要stopword.txt, pos.txt, neg.txt
用所有数据集建立TFIDF模型和MultinomialNB分类模型，并绘制学习曲线
建立后的模型
TFIDF.model https://xiaojiowo.coding.net/s/3e20858e-cdfa-4f9a-823e-7b1c74c8d86b
bayes.model https://xiaojiowo.coding.net/s/741af8fb-c289-425c-ba36-1df3a719a385
'''
import jieba
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

pos_path = 'pos.txt'
neg_path = 'neg.txt'
stopwords_path = 'stopword.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]


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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


pos_cut = cut_words(pos)
neg_cut = cut_words(neg)

TFIDF_model = TfidfVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}',
                              ngram_range=(1, 4),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1,
                              stop_words=None)

label = [1] * 16548 + [0] * 18581
data_all = pos_cut + neg_cut  # 16548 18581
len_pos = len(pos_cut)

TFIDF_model.fit(data_all)
# print(len(data_all))  # 35129
# print(len(label))  # 35129

TFIDF_model.fit(data_all)
data_train = TFIDF_model.transform(data_all)
model = MultinomialNB()
model.fit(data_train, label)


def save_mod():
    joblib.dump(TFIDF_model, 'TFIDF.model')
    joblib.dump(model, 'bayes.model')


save_mod()

alpha = [0.5]
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
for i in range(len(alpha)):
    title = "Learning Curves (alpha=" + str(alpha[i]) + ")"
    model = MultinomialNB(alpha=alpha[i], class_prior=None, fit_prior=False)
    plot_learning_curve(model, title, data_train, label, ylim=(0, 1.01), cv=cv, n_jobs=-1)
    plt.show()

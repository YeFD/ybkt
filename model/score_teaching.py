# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 13:35
# @File    : score_teaching.py
import csv
import jieba
import time
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

start = time.process_time()

pos_path = 'pos.txt'
neg_path = 'neg.txt'
stopwords_path = 'stopword.txt'
TFIDF_model = joblib.load('TFIDF.model')
path = 'output.csv'

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


teaching_list = []
teaching_label = []
with open(path, encoding='utf-8') as output:
    csvReader = csv.reader(output)
    for row in csvReader:
        # sentence = del_stopwords(row[0])
        teaching_list += [row[0]]
        if float(row[1]) >= 3.0:
            teaching_label += [1]
        else:
            teaching_label += [0]
# pos neg 21861 357
sentences = cut_words(teaching_list)
# print(len(sentences))  # 22218
# print(len(label))  # 22218
result_data = []
result_label = []
for i in range(0, len(teaching_label)):
    if len(sentences[i]) >= 5:
        result_data += [sentences[i]]
        result_label += [teaching_label[i]]

pos_cut = cut_words(pos)
neg_cut = cut_words(neg)

TFIDF_model = TfidfVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}',
                              ngram_range=(1, 4),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1,
                              stop_words=None)

data_all = pos_cut + neg_cut + result_data  # 16548 18581
label = [1] * 16548 + [0] * 18581 + result_label

TFIDF_model.fit(data_all)
data_train = TFIDF_model.transform(data_all)
train_data, test_data, train_label, test_label = train_test_split(data_train, label, test_size=0.3, random_state=2)


MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # 默认
model = MultinomialNB()

model.fit(train_data, train_label)
predict = model.predict(test_data)
pos_worse = 0
neg_worse = 0
for i in range(0, len(predict)):
    if predict[i] != test_label[i]:
        if predict[i] == 1:
            pos_worse += 1
        else:
            neg_worse += 1
pos_count = 0
neg_count = 1
for i in range(len(test_label)):
    if test_label[i] == 1:
        pos_count += 1
    else:
        neg_count += 1
acc = 1 - (pos_worse + neg_worse) / len(test_label)
print(acc)  # 0.9023423636578729
print(pos_worse, neg_worse)  # 1563 88
end = time.process_time()  # time= 53.1875
P = pos_count
N = neg_count
FP = pos_worse
FN = neg_worse
TP = P - FN
TN = N - FP
acc = (TP + TN) / (P + N)
pre = TP / (TP + FP)
print(TP, P)
recall = TP / P
F1 = 2 * (pre * recall) / (pre + recall)
print("total=", P + N)
print("acc=", acc)
print("pre=", pre)
print("recall=", recall)
print("F1=", F1)
'''
0.9058322489057139
0.9058322489057139
1516 76
11160 11236
total= 16907
acc= 0.905837818654995
pre= 0.8804039129062796
recall= 0.9932360270558918
F1= 0.9334225493476079
'''

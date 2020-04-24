# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:21
# @File    : score_TFIDF-SDG.py
'''
求TFIDF+SDG的默认ACC
'''
import jieba
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

stopwords_path = 'stopword.txt'
pos_path = 'pos.txt'
neg_path = 'neg.txt'
TFIDF_model = joblib.load('TFIDF.model')

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


pos_cut = cut_words(pos)
neg_cut = cut_words(neg)

label = [1] * 16548 + [0] * 18581
data_all = pos_cut + neg_cut  # 16548 18581
data_train = TFIDF_model.transform(data_all)


train_data, test_data, train_label, test_label = train_test_split(data_train, label, test_size=0.3, random_state=2)

model = SGDClassifier()

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
print(pos_worse, neg_worse)
# pos_count = 16548
# neg_count = 18581
print(pos_worse, neg_worse, pos_count, neg_count)
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
271 649
271 649 4953 5587
4304 4953
total= 10540
acc= 0.9127134724857685
pre= 0.9407650273224044
recall= 0.8689683020391682
F1= 0.9034424853064651
'''

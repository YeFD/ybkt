# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:18
# @File    : score_SnowNLP.py
'''

'''
from snownlp import SnowNLP
from sklearn.model_selection import train_test_split

pos_path = 'pos.txt'
neg_path = 'neg.txt'
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data_all = pos + neg
label = [1]*16548 + [0]*18581
train_data, test_data, train_label, test_label = train_test_split(data_all, label, test_size=0.3, random_state=2)


def get_score(sentence):
    return SnowNLP(sentence).sentiments


def get_predict(sentence_list):
    predict_list = []
    for sentence in sentence_list:
        # if sentence == "":
        #     predict_list.append(0)
        #     continue
        if get_score(sentence) >= 0.5:
            predict_list.append(1)
        else:
            predict_list.append(0)
    return predict_list


def get_acc(pre, y):
    l = len(pre)
    count = 0
    for i in range(l):
        if pre[i] == y[i]:
            count += 1
        else:
            continue
    return count/l


label = test_label
predict = get_predict(test_data)
print(get_acc(predict, label))  # 0.8115882352941176
pos_worse = 0
neg_worse = 0
pos_count = 0
neg_count = 0
for i in range(len(label)):
    if label[i] == 1:
        if predict[i] == 0:
            pos_worse += 1
        pos_count += 1
    else:
        if predict[i] == 1:
            neg_worse += 1
        neg_count += 1
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
0.852263023057216
931 626 4953 5586
4327 4953
total= 10539
acc= 0.852263023057216
pre= 0.8229364777481932
recall= 0.8736119523521099
F1= 0.8475173832141808
'''
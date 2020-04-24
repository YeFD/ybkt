# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:18
# @File    : score_BaiduAPI.py
'''
读取csv，计算各评价标准
'''
import csv
pos_worse = 0
neg_worse = 0
pos_count = 0
neg_count = 0
with open("BaiduAPI.csv", encoding='utf-8') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        # print(row)
        if row[1] == '1':
            if row[3] == '0':
                pos_worse += 1
            pos_count += 1
        else:
            if row[3] == '1':
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
2429 2549 16547 18578
13998 16547
total= 35125
acc= 0.8582775800711744
pre= 0.852133682352225
recall= 0.8459539493563788
F1= 0.8490325711166373
'''
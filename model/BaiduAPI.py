# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:19
# @File    : BaiduAPI.py
'''
需要pos.txt, neg.txt
调用百度文本情感分析API 分析语料, 结果保存在BaiduAPI.csv
'''
import sys
import json
import base64
import time
import requests
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
import csv
from urllib.parse import urlencode
from urllib.parse import quote_plus

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
COMMENT_TAG_URL = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify"
API_KEY = 'cgR24pHKGuR5DV36oudDFm8L'
SECRET_KEY = 'abOHGrvcTTm22wMCsjxSuHVf0fhoqPaK '


def fetch_token():
    host = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=cgR24pHKGuR5DV36oudDFm8L&client_secret=abOHGrvcTTm22wMCsjxSuHVf0fhoqPaK"
    response = requests.get(host)
    if response:
        return response.json()['access_token']


def make_request(url, comment):
    response = request(url, json.dumps(
        {
            "text": comment,
        }))

    data = json.loads(response)

    time.sleep(0.5)
    if "error_code" not in data or data["error_code"] == 0:
        for item in data["items"]:
            # print(item["positive_prob"])
            return item["positive_prob"]
    else:
        print(response)


def request(url, data):
    try:
        req = Request(url, data.encode('utf-8'))
        has_error = False
        try:
            f = urlopen(req)
            result_str = f.read()
            print(result_str, type(result_str))
            result_str = result_str.decode('utf-8', 'ignore')
            f.close()
            return result_str
        except URLError as err:
            print(err)
    except:
        request(url, data)


token = fetch_token()
print(token)
url = COMMENT_TAG_URL + "?access_token=" + token
# comment1 = "手机已经收到，非常完美超出自己的想象，外观惊艳 黑色高端加外形时尚融为一体比较喜欢的类型。系统流畅优化的很好，操作界面简洁大方好上手。电池用量很满意，快充很不错。相机拍人拍物都美。总而言之一句话很喜欢的宝贝。"
# make_request(url, comment1)

pos_path = 'pos.txt'
neg_path = 'neg.txt'

pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
comments = pos + neg
label = [1] * 16548 + [0] * 18581  # 35129
predict = []
csvWriter = csv.writer(open("BaiduAPI.csv", 'a', newline='', encoding='utf-8'))
pos_worse = 0
neg_worse = 0
pos_count = 0
neg_count = 0
for i in range(33934, len(comments)):
    if sys.getsizeof(comments[i]) > 2048:
        continue
    score = make_request(url, comments[i])
    print(i, score)
    if score > 0.5:
        temp = 1
        pos_count += 1
    else:
        temp = 0
        neg_count += 1
    csvWriter.writerow([comments[i], label[i], score, temp])
    if temp != label[i]:
        if temp == 1:
            pos_worse += 1
        else:
            neg_worse += 1
    predict.append(temp)

# acc = 1 - (pos_worse + neg_worse) / (pos_count + neg_count)
# print("acc=", acc)
# print(pos_worse, neg_worse, pos_count, neg_count)

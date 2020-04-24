# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 13:24
# @File    : w2v_model.py
'''
单文件太大，拆分文件需要合并
Word60.model.syn1neg.npy https://xiaojiowo.coding.net/s/17db0a65-6dd0-4624-90e7-63f7cb135474
Word60.model.syn0.npy https://xiaojiowo.coding.net/s/d65f2a81-0b2d-4e69-a98b-c590b0daeda9
Word60.model https://xiaojiowo.coding.net/s/2252ac49-de5f-4004-96b9-84f8066c467f
'''
from gensim.models import Word2Vec
import joblib
w2v_model = Word2Vec.load('Word60.model')
joblib.dump(w2v_model, 'w2v.model')

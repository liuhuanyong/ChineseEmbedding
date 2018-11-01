#!/usr/bin/env python3
# coding: utf-8
# File: test_vector.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-1

import os
import gensim
from gensim.models import word2vec
from sklearn.decomposition import PCA
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

class TrainVector:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.token_embedding = os.path.join(cur, 'model/token_vec_300.bin')
        self.postag_embedding = os.path.join(cur, 'model/postag_vec_30.bin')
        self.dep_embedding = os.path.join(cur, 'model/dep_vec_10.bin')
        self.pinyin_embedding = os.path.join(cur, 'model/pinyin_vec_300.bin')
        self.word_embedding = os.path.join(cur, 'model/word_vec_300.bin')

    '''对训练好的模型进行测试'''
    def test_model(self, embedding_path):
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        while (1):
            wd = input('enter an word to search:')
            result = model.most_similar(wd)
            for res in result:
                print(res)
        return

    '''训练主函数'''
    def test_main(self):
        # 测试字向量
        # self.test_model(self.token_embedding)
        # 测试依存向量
        # self.test_model(self.dep_embedding)
        # 测试拼音向量
        # self.test_model(self.pinyin_embedding)
        # 测试词性向量
        # self.test_model(self.postag_embedding)
        # 测试词向量
        self.test_model(self.word_embedding)
        return

if __name__ == '__main__':
    handler = TrainVector()
    handler.test_main()


#!/usr/bin/env python3
# coding: utf-8
# File: train.py.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-26

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
        # 训练语料所在目录
        self.token_filepath = os.path.join(cur, 'train_data/token_train.txt')
        self.pinyin_filepath = os.path.join(cur, 'train_data/pinyin_train.txt')
        self.postag_filepath = os.path.join(cur, 'train_data/postag_train.txt')
        self.dep_filepath = os.path.join(cur, 'train_data/dep_train.txt')
        self.word_filepath = os.path.join(cur, 'train_data/word_train.txt')

        # 向量文件所在目录
        self.token_embedding = os.path.join(cur, 'model/token_vec_300.bin')
        self.postag_embedding = os.path.join(cur, 'model/postag_vec_30.bin')
        self.dep_embedding = os.path.join(cur, 'model/dep_vec_10.bin')
        self.pinyin_embedding = os.path.join(cur, 'model/pinyin_vec_300.bin')
        self.word_embedding = os.path.join(cur, 'model/word_vec_300.bin')

        #向量大小设置
        self.token_size = 300
        self.pinyin_size = 300
        self.dep_size = 10
        self.postag_size = 30
        self.word_size = 300


    '''基于gensimx训练字符向量,拼音向量,词性向量'''
    def train_vector(self, train_path, embedding_path, embedding_size):
        sentences = word2vec.Text8Corpus(train_path)  # 加载分词语料
        model = word2vec.Word2Vec(sentences, size=embedding_size, window=5, min_count=5)  # 训练skip-gram模型,默认window=5
        model.wv.save_word2vec_format(embedding_path, binary=False)

    '''基于特征共现+pca降维的依存向训练'''
    def train_dep_vector(self, train_path, embedding_path, embedding_size):
        f_embedding = open(embedding_path, 'w+')
        deps = ['SBV', 'COO', 'ATT', 'VOB', 'FOB', 'IOB', 'POB', 'RAD', 'ADV', 'DBL', 'CMP', 'WP', 'HED', 'LAD']
        weight_matrix = []
        for dep in deps:
            print(dep)
            weights = []
            for line in open(train_path):
                line = line.strip().split('\t')
                dep_dict = {i.split('@')[0]:int(i.split('@')[1]) for i in line[1].split(';')}
                sum_tf = sum(dep_dict.values())
                dep_dict = {key:round(value/sum_tf,10) for key, value in dep_dict.items()}
                weight = dep_dict.get(dep, 0.0)
                weights.append(str(weight))
            weight_matrix.append(weights)
        weight_matrix = np.array(weight_matrix)
        pca = PCA(n_components = embedding_size)
        low_embedding = pca.fit_transform(weight_matrix)
        for index, vecs in enumerate(low_embedding):
            dep = deps[index]
            vec = ' '.join([str(vec) for vec in vecs])
            f_embedding.write(dep + ' ' + vec + '\n')
        f_embedding.close()

    '''训练主函数'''
    def train_main(self):
        #训练依存向量
        self.train_dep_vector(self.dep_filepath, self.dep_embedding, self.dep_size)
        #训练汉字字向量
        self.train_vector(self.token_filepath, self.token_embedding, self.token_size)
        #训练汉语词性向量
        self.train_vector(self.postag_filepath, self.postag_embedding, self.postag_size)
        #训练汉语词向量
        self.train_vector(self.word_filepath, self.word_embedding, self.word_size)
        # 训练汉语拼音向量
        self.train_vector(self.pinyin_filepath, self.pinyin_embedding, self.pinyin_size)
        return

if __name__ == '__main__':
    handler = TrainVector()
    handler.train_main()


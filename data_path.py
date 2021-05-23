#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from os import path

# 3.1 原始数据
passages_multi_sentences = 'lab2-data/passages_multi_sentences.json'
new_test = 'lab2-data/new_test.json'
train = 'lab2-data/train.json'

# 停用词表
stopwords_file = 'stopwords(new).txt'

# 3.1 生成数据
passages_segment = 'lab2-data/preprocessed/passages_seg.json'
index_dir = 'lab2-data/preprocessed/index'
corpus = 'lab2-data/preprocessed/corpus.txt'
BM25Model = 'lab2-data/preprocessed/bm25.pkl'
dic_path = 'lab2-data/preprocessed/dic.bin'

# 3.2 原始数据
train_questions = 'lab2-data/question_classification/train_questions.txt'
test_questions = 'lab2-data/question_classification/test_questions.txt'

# 3.2 生成数据
train_questions_seg = 'lab2-data/question_classification/train_questions_seg.txt'
test_questions_seg = 'lab2-data/question_classification/test_questions_seg.txt'
svm_model = 'lab2-data/question_classification/svm_model.model'

# 3.3 生成数据
answer_sentence = 'lab2-data/answer_sentence_selection/answer_sentence.json'
answer_sentence_test = 'lab2-data/answer_sentence_selection/answer_sentence_test.json'
select_sentence = 'lab2-data/answer_sentence_selection/select_sentence.json'
sentences = 'lab2-data/answer_sentence_selection/sentences.txt'
w2v = 'lab2-data/answer_sentence_selection/word2vec.model'
answer_train = 'lab2-data/answer_sentence_selection/train.json'
answer_test = 'lab2-data/answer_sentence_selection/test.json'
svm_rank_feature_train = 'lab2-data/answer_sentence_selection/svm_rank_feature_train.txt'
svm_rank_feature_test = 'lab2-data/answer_sentence_selection/svm_rank_feature_test.txt'
prediction = 'lab2-data/answer_sentence_selection/predictions'

# 3.4 生成数据
train_output = 'lab2-data/answer_span_selection/train.output.txt'
test_output = 'lab2-data/answer_span_selection/test_output.txt'
user_dict = 'lab2-data/answer_span_selection/user_dict.txt'
test_answer='lab2-data/answer_span_selection/test_answer.json'

# 日志记录
log_file = 'log.txt'
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', \
                    datefmt='%a, %d %b %Y %H:%M:%S')

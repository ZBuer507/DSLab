#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jieba
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def questions_seg():
    result = []
    with open('lab2-data/question_classification/train_questions.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            result.append("{}\t{}\n".format(attr[0], ' '.join(jieba.cut(attr[1]))))
    with open('lab2-data/question_classification/train_questions_seg.txt', 'w', encoding='utf-8') as f:
        f.writelines(result)
    
    result = []
    with open('lab2-data/question_classification/test_questions.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            result.append("{}\t{}\n".format(attr[0], ' '.join(jieba.cut(attr[1]))))
    with open('lab2-data/question_classification/test_questions_seg.txt', 'w', encoding='utf-8') as f:
        f.writelines(result)


def load_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open('lab2-data/question_classification/train_questions_seg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_train.append(attr[1])
            y_train.append(attr[0].split('_')[0])
    with open('lab2-data/question_classification/test_questions_seg.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            attr = line.strip().split('\t')
            x_test.append(attr[1])
            y_test.append(attr[0].split('_')[0])
    return x_train, y_train, x_test, y_test


def logistic_regression():
    x_train, y_train, x_test, y_test = load_data()
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    #tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)
    lr = LogisticRegression(C=8000, solver='liblinear', multi_class='ovr')
    lr.fit(train_data, y_train)
    print(lr.score(test_data, y_test))


def support_vector():
    x_train, y_train, x_test, y_test = load_data()
    tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    #tv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_data = tv.fit_transform(x_train)
    test_data = tv.transform(x_test)
    clf = SVC(C=2.1, kernel="sigmoid")
    clf.fit(train_data, y_train)
    print(clf.score(test_data, y_test))


if __name__ == '__main__':
    #questions_seg()
    logistic_regression()
    #support_vector()

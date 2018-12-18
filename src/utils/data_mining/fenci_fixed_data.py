# -*- coding: utf-8 -*-
#
# require package list:
#	- pip install jieba
#	- pip install scikit-learn

import os
import random
from collections import Counter
import numpy as np
import pickle

import jieba
import jieba.analyse

from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib

CONTENT_TYPE = {
    'yl': '娱乐',
    'sports': '体育',
    'finance': '财经',
    'IT': '科技',
    'mil': '军事',
    'house': '房产',
    'jk': '健康',
    'energy': '能源',
    'edu': '教育',
    'wh':'文化'
}

test_word_file_name_list = []

def readFile(new_foler_path, file):
    '''
    读取文件内容
    '''
    file_path = os.path.join(new_foler_path, file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError as e:
        print('Read {0} unicode decode error'.format(file_path))

    return None

def writeFile(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(content))
    except UnicodeDecodeError as e:
        print('Write {0} unicode decode error'.format(file_path))
    return None

"""
    Parameters:
    folder_path - 文本存放路径
    word_list - 分词结果列表
    class_list - 文本类型列表
    train_words_list
    test_words_list
    train_feature_list
    test_feature_list
    all_words_list
    全部样本 - 分词（筛选，关键词-类别）- 向量化 - 训练集、测试集分开 - 训练 - 评估
"""
def TextProcessing(folder_path, stop_words_path, user_dict_path, key_words_path, test_size = 0.5):
    class_list = []
    all_word_list = []
    folder_list = os.listdir(folder_path)
    all_word_set = set([])
    all_word_dict = {}
    
    train_word_list = []
    train_class_list = []

    test_word_list = []
    test_class_list = []
    
    key_words_txt = "key_words.txt"
    
    for folder in folder_list:
        if os.path.isdir(os.path.join(folder_path,folder)):
            print("------&&&&-----")
            print(folder)

            new_foler_path = os.path.join(folder_path,folder)
            files = os.listdir(new_foler_path)
            count = 1

            maxLen = len(files)
            
            for file in files:

                # 读取文件
                content = readFile(new_foler_path, file)
                if content is None:
                    continue

                jieba.analyse.set_stop_words(stop_words_path)
                jieba.load_userdict(user_dict_path)
                jieba.suggest_freq(("道", "题"), tune = True)
                jieba.suggest_freq(("有", "没有"),tune = True)
                jieba.suggest_freq(("级", "新生"), tune = True)
                jieba.suggest_freq(("中", "四人"), tune = True)
                jieba.suggest_freq(("曝", "与"), tune = True)
                jieba.suggest_freq(("身", "披"), tune = True)
                jieba.suggest_freq(("爱滋病", "患"), tune = True)
                jieba.suggest_freq(("平均", "收入"), tune = True)		
                
                #筛选名词、机构团体名
                content_cut = jieba.analyse.extract_tags(content, topK = 30, allowPOS = {'n', 'nt'})
               
               #maxLen * test_size
                if count < maxLen * test_size:
                    train_word_list.append(Convert2Str(content_cut))
                    train_class_list.append(CONTENT_TYPE[folder])
                # elif count > 10 and count < 20:
                else:
                    test_word_file_name_list.append(new_foler_path + '/' + file)
                    test_word_list.append(Convert2Str(content_cut))
                    test_class_list.append(CONTENT_TYPE[folder])

                count = count + 1
                    
                #write the top_k_word into document
            
                '''with open(key_words_path, 'a', encoding='utf-8') as top_k_word:
                    top_k_word.write(new_foler_path + '/' + file + '\n')
                    top_k_word.write(str(content_cut))
                    top_k_word.write("\n---***分割线***---")
                    top_k_word.write("\n")
'''
    saveListToFile(train_word_list, 'train_word_list.txt')
    saveListToFile(train_class_list, 'train_class_list.txt')
    saveListToFile(test_word_list, 'test_word_list.txt')
    saveListToFile(test_class_list, 'test_class_list.txt')
    return None

# save list to file
def saveListToFile(l, file):
    with open(file, "wb") as fp:
        pickle.dump(l, fp)

# read list from file
def readListFromFile(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

#将逗号分隔的中文改为用空格分隔连接的字符串
def Convert2Str(iter):
    target_str = ''
    for item in iter:
        target_str = target_str + str(item)
        target_str = target_str + ' '
    return  target_str

def loadData(folder_path, infile):
    f = readFile(folder_path,infile)
    f_split = f[1: len(f) - 1].split(',')
    f_list = list(f_split)
    
    return f_list

if __name__ == '__main__':
    if os.path.exists('train_word_list.txt') == False:
        TextProcessing('./data/source-data','./stop_words.txt', 'user_dict.txt', './key_words.txt', test_size = 0.5)
    #train_word_list, train_class_list, test_word_list, test_class_list

    test_word_list = readListFromFile('test_word_list.txt')
    test_class_list = readListFromFile('test_class_list.txt')

    #train_word_lsit, train_class_list
    train_word_list = readListFromFile('train_word_list.txt')
    train_class_list = readListFromFile('train_class_list.txt')


    #贝叶斯
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_word_list)
    X_test_counts = count_vect.transform(test_word_list)
    print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_class_list)
    if os.path.exists('NB.model') == False:
        joblib.dump(clf, 'NB.model', compress=3)

    clf = joblib.load('NB.model')
    predicted = clf.predict(X_test_counts)

    print("Bayes_Avg_Precision: ",np.mean(predicted==test_class_list))
    print(classification_report(test_class_list, predicted, target_names = CONTENT_TYPE.values()))
    print("Bayes confusion matrix:")
    print(metrics.confusion_matrix(test_class_list, predicted))
    
    print("--------*****---------")

    #SVM
    svm_clf = Pipeline([
                        ('vect',CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',SGDClassifier(loss='hinge',penalty='l2',
                                             alpha =1e-3,random_state=42,
                                             max_iter=5,tol=None)),
                        ])
    svm_clf.fit(train_word_list, train_class_list)
    if os.path.exists('SVM.model') == False:
        joblib.dump(clf, 'SVM.model', compress=3)

    clf_svm = joblib.load('SVM.model')
    predicted = svm_clf.predict(test_word_list)
    print("SVM_Avg_Precision:",np.mean(predicted == test_class_list))
    print(classification_report(test_class_list, predicted, target_names = CONTENT_TYPE.values()))
    print("SVM confusion matrix:")
    print(metrics.confusion_matrix(test_class_list, predicted))

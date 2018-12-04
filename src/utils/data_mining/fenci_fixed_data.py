# -*- coding: utf-8 -*-
#
# require package list:
#	- pip install jieba
#	- pip install scikit-learn

import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import os
import random
from collections import Counter
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

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

def TextProcessing(folder_path, key_words_path, test_size = 0.5):
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
            for file in files:
                with open(os.path.join(new_foler_path,file),'r',encoding='utf-8') as f:
                    content = f.read()
                    jieba.analyse.set_stop_words(key_words_path)
                    #筛选名词、人名、地名、机构团体名、动词
                    content_cut = jieba.analyse.extract_tags(content, topK = 30, allowPOS = {'n', 'nr', 'ns', 'nt', 'v'})
                    #content_cut = jieba.analyse.extract_tags(content, topK = 30)
                    if count < len(files)/2:
                        train_word_list.append(Convert2Str(content_cut))
                        train_class_list.append(CONTENT_TYPE[folder])
                    else:
                        test_word_list.append(Convert2Str(content_cut))
                        test_class_list.append(CONTENT_TYPE[folder])
                    count = count + 1
                    all_word_list.append(Convert2Str(content_cut))
                    class_list.append(CONTENT_TYPE[folder])
                    
                    #write the top_k_word into document
            
                    """with open(os.path.join(key_words_path, key_words_txt), 'a', encoding = 'utf-8') as top_k_word:
                        top_k_word.write(str(content_cut))
                        top_k_word.write("---***分割线***---")
                    top_k_word.write("\n")"""

    return all_word_list, train_word_list, train_class_list, test_word_list, test_class_list

#将逗号分隔的中文改为用空格分隔连接的字符串
def Convert2Str(iter):
    target_str = ''
    for item in iter:
        target_str = target_str + str(item)
        target_str = target_str + ' '
    return  target_str

if __name__ == '__main__':
    all_word_list, train_word_list, train_class_list, test_word_list, test_class_list = TextProcessing('./data/source-data', \
			'./data/keywords/stop_words.txt', \
			test_size = 0.5)
    
    #贝叶斯 -- 0.8976
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_word_list)
    X_test_counts = count_vect.transform(test_word_list)
    print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_class_list)
    predicted = clf.predict(X_test_counts)
    print("Bayes: ",np.mean(predicted==test_class_list))

    #SVM -- 0.9007
    svm_clf = Pipeline([
                       ('vect',CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',SGDClassifier(loss='hinge',penalty='l2',
                                            alpha =1e-3,random_state=42,
                                            max_iter=5,tol=None)),
                        ])
    svm_clf.fit(train_word_list, train_class_list)
    predicted = svm_clf.predict(test_word_list)
    print("SVM:",np.mean(predicted == test_class_list))


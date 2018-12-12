# -*- coding: utf-8 -*-
#
# require package list:
#	- pip install jieba
#	- pip install scikit-learn

import os
import random
from collections import Counter
import numpy as np

import jieba
import jieba.analyse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


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

            # 测试环境下，默认读取 100 篇以便快速出结果
            read_file_num = 0
            MAX_FILE_NUM = 100
            maxLen = MAX_FILE_NUM # len(files)
            
            for file in files:

                # break
                # 读取超过 100 篇就 break
                if read_file_num > 100:
                    read_file_num = 0
                    print("INFO: read {0} finish".format(folder))
                    break

                read_file_num = read_file_num + 1

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


                #筛选名词、机构团体名、动词
                content_cut = jieba.analyse.extract_tags(content, topK = 30, allowPOS = {'n', 'nt', 'v'})
               
                if count < maxLen * test_size:
                    train_word_list.append(Convert2Str(content_cut))
                    train_class_list.append(CONTENT_TYPE[folder])
                else:
                    test_word_list.append(Convert2Str(content_cut))
                    test_class_list.append(CONTENT_TYPE[folder])

                count = count + 1
                all_word_list.append(Convert2Str(content_cut))
                class_list.append(CONTENT_TYPE[folder])
                    
                #write the top_k_word into document
            
                with open(key_words_path, 'a', encoding='utf-8') as top_k_word:
                    top_k_word.write(new_foler_path + '/' + file + '\n')
                    top_k_word.write(str(content_cut))
                    top_k_word.write("---***分割线***---")
                    top_k_word.write("\n")

    return all_word_list, train_word_list, train_class_list, test_word_list, test_class_list

#将逗号分隔的中文改为用空格分隔连接的字符串
def Convert2Str(iter):
    target_str = ''
    for item in iter:
        target_str = target_str + str(item)
        target_str = target_str + ' '
    return  target_str

if __name__ == '__main__':
    all_word_list, train_word_list, train_class_list, test_word_list, test_class_list = TextProcessing('./data/source-data','./stop_words.txt', 'user_dict.txt', \
			'./key_words.txt', \
			test_size = 0.5)
    
    #贝叶斯 --
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_word_list)
    X_test_counts = count_vect.transform(test_word_list)
    print(X_train_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_class_list)
    predicted = clf.predict(X_test_counts)
    print("Bayes_Avg_Precision: ",np.mean(predicted==test_class_list))
    print(classification_report(test_class_list, predicted, target_names = CONTENT_TYPE.values()))
    print("Bayes confusion matrix:")
    print(metrics.confusion_matrix(test_class_list, predicted))
    
    print("--------*****---------")

    #SVM --
    svm_clf = Pipeline([
                       ('vect',CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',SGDClassifier(loss='hinge',penalty='l2',
                                            alpha =1e-3,random_state=42,
                                            max_iter=5,tol=None)),
                        ])
    svm_clf.fit(train_word_list, train_class_list)
    predicted = svm_clf.predict(test_word_list)
    print("SVM_Avg_Precision:",np.mean(predicted == test_class_list))
    print(classification_report(test_class_list, predicted, target_names = CONTENT_TYPE.values()))
    print("SVM confusion matrix:")
    print(metrics.confusion_matrix(test_class_list, predicted))

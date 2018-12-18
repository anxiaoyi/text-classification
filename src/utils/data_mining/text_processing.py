# -*- coding: utf-8 -*-

import os
import jieba
import jieba.analyse


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
                
                if count > 20:
                    break

        #筛选名词、机构团体名
        content_cut = jieba.analyse.extract_tags(content, topK = 30, allowPOS = {'n', 'nt'})
            
            #maxLen * test_size
            if count < 10:
                train_word_list.append(Convert2Str(content_cut))
                train_class_list.append(CONTENT_TYPE[folder])
                elif count > 10 and count < 20:
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
    writeFile('train_word_list.txt', str(train_word_list))
    writeFile('train_class_list.txt', str(train_class_list))
    writeFile('test_word_list.txt', str(test_word_list))
    writeFile('test_class_list.txt', str(test_class_list))
    return None



if __name__ == '__main__':
    if os.path.exists('train_word_list.txt') == False:
        TextProcessing('./data/source-data','./stop_words.txt', 'user_dict.txt', './key_words.txt', test_size = 0.5)
    


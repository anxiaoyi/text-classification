#！/usr/bin/env python

import numpy as np
import math


types = {0:"技术",1:"天气",2:"娱乐",3:"体育",4:"军事"}

# create dict of each type
def createBOW(chaList,types):
    wordBag = dict([])
    for i in range(len(types)):
        wordBag[i] = set(chaList[i])
    return wordBag

# calculate total number of each word in each type, storing the value in matrix
def traingN0(trainList, types):
    bagList = createBOW(trainList, types)
    mat = np.zeros([5,10])
    sumArr = np.zeros([1,5])
    proMat = np.zeros([5,10])
    for i in range(len(types)):
        trainCha = trainList[i]
        bagCha = bagList[i]
        for cha in trainCha:
            if cha in bagCha:
                bagChaList = list(bagCha)
                mat[i,bagChaList.index(cha)] += 1
            else:
                print("the word %s is not in the bag"%cha)
    for i in range(len(types)):
        print(mat[i])
        sumArr[0,i] = 1#sum(mat[i])
    mat += 1.0
    for t in range(len(mat)):
        for r in range(len(mat[t])):
            proMat[t,r] = math.log((mat[t,r]/sumArr[0,t]),3)
    return bagList,proMat


# 计算testEntry中每个词在每类中出现的概率，取最大者最为最终结果
def classifyEntry(dataSet,testEntry,types):
    bagList, proMat = traingN0(dataSet, types)
    pD = np.arange(len(types)).reshape(1,len(types))
    for key in bagList:
        for word in testEntry:
            if word in bagList[key]:
                pD[0,key] += proMat[key, list(bagList[key]).index(word)]
    max = 0.0
    index = 0
    for i in range(len(pD[0])):
        if(max<pD[0,i]):
            max = pD[0,i]
            index = i
    return types[i]
    

trainList = [["计算机","视觉","科技","计算机"],
           ["天气","交通","出行","天气"],
           ["拍戏","电视剧","电影","音乐"],
           ["游泳","滑冰","比赛","游泳","比赛"],
           ["波斯湾","石油","伊拉克","美国","伊拉克"]]

testEntry = ["石油","美国"]

# print(traingN0(trainList,types))
print(classifyEntry(trainList,testEntry,types))
        

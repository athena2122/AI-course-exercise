# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
 This program is to train or generate peom.
'''''''''''''''''''''''''''''''''''''''''''''''''''''
from header import *
import numpy as np


def readpoems(filename):
    poems = []
    with open(filename,"r",encoding="UTF-8")as f:

        for line in f.readlines():  # 每行储存一首诗
            # 数据集中可能不止一个：，限定只切一刀
            title,poem=line.strip().split(':',1)

            poem = poem.replace(' ','') #去除诗中的空格
            ## replace替换函数。
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            ## 去除poem的其它字符

            if len(poem) < 10 or len(poem) > 128:
                continue
            ## 确保poem长度适当

            poem = '[' + poem + ']' #add start and end signs
            poems.append(poem)
            # 将poem加到列表 poems当中

        print("Number of Poems： %d" % len(poems))
        #counting words
        allWords = {}
        # 定义一个空的字典(dictionary)
        # key:为汉字
        # 字典的值value:为该汉字出现的频次
        for poem in poems: # 枚举诗歌列表中所有诗
            for word in poem: # 枚举每首诗中所有的字

                if word not in allWords:
                    allWords[word] = 1  # 假如该汉字还没出现过
                else:
                    allWords[word] += 1
        #'''
        # 删除低频字
        erase = []
        for key in allWords: # 枚举字典allwords中的所有字key
            if allWords[key] < 2:  #假如其频次小于2，加入待删列表
                erase.append(key)
        for key in erase:
            del allWords[key]
        #'''
        # eg:wordPairs:[('你',33),('我',25)]所有出现的汉字按出现频次，由多到少排序
        wordPairs = sorted(allWords.items(), key = lambda x: x[1],reverse=True)


        # eg:words:('你','我')
        # a:(33,25)
        words, a= zip(*wordPairs)

        words += (" ", )           # 将空格加在最后。
        #若len(words)=3 zip((你，我，他)，（0，1，2）)后变为[（你，0），（我，1），（他，2）]
        wordToID = dict(zip(words, range(len(words)))) #word to ID
                                                       # 把每个汉字赋予一个数。
        #输入：A：你-》0
        wordTOIDFun = lambda A: wordToID.get(A, len(words))  # 转化成函数，
                                                            # 输入汉字，输出它对应的数。
        ## 将所有的诗转化成向量，其中一首诗中，
        ## eg:[[1 3 2 4], [2 3 4 1], [1 5 6 2]]
        poemsVector = [([wordTOIDFun(word) for word in poem]) for poem in poems] # poem to vector
        #padding length to batchMaxLength
        batchNum  = (len(poemsVector) - 1) // batchSize  # // 除法将余数去掉，如10//3 = 3



        X = []
        Y = []
        #create batch
        for i in range(batchNum):
            batch = poemsVector[i * batchSize: (i + 1) * batchSize]
            # batch 储存了 batchsize诗的向量

            maxLength = max([len(vector) for vector in batch])
            # 得到一个batch其中一首最长的诗的长度

            temp = np.full((batchSize, maxLength), wordTOIDFun(" "), np.int32)
            #将temp初始化成batchsize * maxlength的矩阵，其中矩阵元素初始值皆为空格对应的ID。

            for j in range(batchSize):
                temp[j, :len(batch[j])] = batch[j]
            #将temp 储存了一批诗对应的矩阵
            X.append(temp)  # 把这个矩阵放入列表X中，
            temp2 = np.copy(temp)
            # eg:temp = [(白日依山进),()]->temp2 [(日依山进进), ()]
            temp2[:, :-1] = temp[:, 1:]#将诗向前挪一个字的位置对应的向量放入Y中。
            Y.append(temp2)
        return X, Y, len(words) + 1, wordToID, words
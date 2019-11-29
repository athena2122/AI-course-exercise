import datahelper
import numpy as np
from header import *
#X:[batchnum,batchsize,dim]
X,Y,wordnum,wordToID,words = datahelper.readpoems(trainPoems)
Xs = ""
Ys = ""
#从三维列表中取出
for i in X[0][0]:
    Xs += words[i]
for i in X[0][0]:
    Ys += words[i]
print('第一批诗的第一首:',Xs)
print('第一批诗的第一首转化成数字后的形式:',X[0][0][:len(Xs.strip())]) # 看到第一批诗（16首）转化成数字后的形式
print('第一批诗的第一首对应的输出:',Ys)
print('第一批诗的第一首对应的输出转化成数字后的形式:',Y[0][0][:len(Ys.strip())]) # 看到第一批诗（16首）转化成数字后的形式

import tensorflow as tf
import numpy as np
from header import *
def buildModel(wordNum, gtX, hidden_units = 128, layers = 2):
    """build rnn"""

    with tf.variable_scope("embedding"): #embedding
           embedding = tf.get_variable("embedding", [wordNum, hidden_units], dtype = tf.float32)
           # embedding 为 n * m维的矩阵，其中n为涉及的汉字的数目，m为隐藏LSTM单元的数目
           # 将汉字用hidden_units维向量重新表征，训练RNN的其中一个目的就是为了找到embedding这个矩
           #将输入buildmodel的变量gtX（整数表征），通过内嵌表embedding，转变成向量表征变量inputbatch。
           inputbatch = tf.nn.embedding_lookup(embedding, gtX)
           # 将16首诗转化为向量表达。
           # embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。比如说，
           #ids=[1,3,2],就是返回params中第1,3,2行。返回结果为由params的1,3,2行组成的tensor。
           # gtx: batchsize * maxLength, embedding (wordnum*hidden)
           # inputbatch: batchsize * maxLength * hidden
    #BasicLSTMCell：搭建LSTM长短记忆基本模块
    basicCell = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple = True)
    #将LSTM单元垒起来
    stackCell = tf.contrib.rnn.MultiRNNCell([basicCell] * layers)
    #构建动态RNN。
    initState = stackCell.zero_state(np.shape(gtX)[0], tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(stackCell, inputbatch, initial_state = initState)
    outputs = tf.reshape(outputs, [-1, hidden_units])

    with tf.variable_scope("softmax"):
        w = tf.get_variable("w", [hidden_units, wordNum])
        b = tf.get_variable("b", [wordNum])
        logits = tf.matmul(outputs, w) + b

    probs = tf.nn.softmax(logits)
    return logits, probs, stackCell, initState, finalState
def train(X,Y,wordNum,reload="True"):
    gtX = tf.placeholder(tf.int32, shape=[batchSize, None])
    gtY = tf.placeholder(tf.int32, shape=[batchSize, None])
    logits,probs,a,b,c = buildModel(wordNum,gtX)
    targets = tf.reshape(gtY,[-1])#gtY[16,122]
    #loss
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targets],
                                                              [tf.ones_like(targets,dtype=tf.float32)],wordNum)
    cost = tf.reduce_mean(loss)
    tvars = tf.trainable_variables()#获得可训练的变量
    #控制梯度，使得整体梯度的范数平方和不超过5，超过部分将变成5
    grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),5)
    learningRate = learningRateBase
    #构建训练器
    trainOP = tf.train.AdamOptimizer(learningRate).apply_gradients(zip(grads,tvars))
    globalStep = 0
    init_op = tf.global_variables_initializer()
    with tf.Session()as sess:
        sess.run(init_op)
        #保存模型
        saver = tf.train.Saver()
        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            #if have checkPoint,restore checkPoint
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess,checkPoint.model_checkpoint_path)
                print("restore %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkPoint found")
        for epoch in range(epochNum):
            if globalStep % learningRateDecreaseStep==0:
                learningRate = learningRateBase*(0.95**epoch)
            epochSteps = len(X)
            for step,(x,y) in enumerate(zip(X,Y)):
                globalStep = epoch*epochSteps+step
                a,loss = sess.run([trainOP,cost],feed_dict={gtX:x,gtY:y})
                print("epoch:%d steps:%d/%d loss:%3f" % (epoch,step,epochSteps,loss))
                if globalStep%1000==0:
                    print("save model...")
                    saver.save(sess,checkpointsPath,epoch)
def probsToWord(weights,words):
    t = np.cumsum(weights)
    s = np.sum(weights)
    coff = np.random.rand(1)
    #由于t是累积分布，因此5667个数从小到大，此函数返回coff*s排在第几
    index = int(np.searchsorted(t,coff*s))
    return words[index]
def test(wordNum,wordToID,words):
    #input
    gtX = tf.placeholder(tf.int32,shape=[1,None])
    logits,probs,stackCell,initState,finalState = buildModel(wordNum,gtX)
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess,checkPoint.model_checkpoint_path)
            print("model restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkPoint found")
        poems = []
        for i in range(generateNum):
            state = sess.run(stackCell.zero_state(1,tf.float32))
            x = np.array([[wordToID['[']]])#开始标记
            probs1,state = sess.run([probs,finalState],feed_dict={gtX:x,initState:state})
            word = probsToWord(probs1,words)
            poem = ''
            while word!= ']' and word!=' ':
                poem+=word
                if word =='。':
                    poem+='\n'
                x = np.array([[wordToID[word]]])
                #print(word)
                probs2,state = sess.run([probs,finalState],feed_dict={gtX:x,initState:state})
                word = probsToWord(probs2, words)
            print(poem)
            poems.append(poem)
        return poems
def testHead(wordNum, wordToID, words, characters):
    gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
    logits, probs, stackCell, initState, finalState = buildModel(wordNum, gtX)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
        # if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")
            exit(0)
        flag = 1
        endSign = {-1: "，", 1: "。"}
        poem = ''
        state = sess.run(stackCell.zero_state(1, tf.float32))
        x = np.array([[wordToID['[']]])

        probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
        for c in characters:
            word = c
            flag = -flag
            while word != ']' and word != '，' and word != '。' and word != ' ':
                # 代码粘贴处10
                poem += word
                x = np.array([[wordToID[word]]])
                probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})

                word = probsToWord(probs2, words)

            poem += endSign[flag]
            # keep the context, state must be updated
            if endSign[flag] == '。':
                probs2, state = sess.run([probs, finalState],
                                         feed_dict={gtX: np.array([[wordToID["。"]]]), initState: state})
                poem += '\n'
            else:
                probs2, state = sess.run([probs, finalState],
                                         feed_dict={gtX: np.array([[wordToID["，"]]]), initState: state})

        print(characters)
        print(poem)
        return poem


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#1.获取数据集(tensorflow中已经进行封装)
train_nums = mnist.train.num_examples
validation_nums = mnist.validation.num_examples
test_nums = mnist.test.num_examples
print("train_nums: %d,validation_nums:%d,test_nums:%d" % (train_nums,validation_nums,test_nums))
#2.获取数据值
train_data = mnist.train.images
val_data = mnist.validation.images
test_data = mnist.test.images
print("训练集数据大小：",train_data.shape)
print("一张图片的形状:",train_data.shape[0])
#3.获取标签值
train_labels = mnist.train.labels
validation_labels = mnist.validation.labels
test_labels = mnist.test.labels
print("训练集标签大小：",train_labels.shape)
print("标签值大小：",train_labels.shape[0])
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])
#3.构建计算图
#inputs:输入该网络层的数据  units:输出的维度大小 activation:激活函数
hidden1 = tf.layers.dense(x,100,activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1,100,activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2,80,activation=tf.nn.relu)
#输出层
y_ = tf.layers.dense(hidden3,10)
#计算损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))

#利用梯度下降进行优化
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
#计算准确率
equal_list = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
# 进行训练
init_op = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 500
with tf.Session()as sess:
    sess.run(init_op)
    for i in range(train_steps):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        loss_val,acc_val,_ = sess.run([loss,accuracy,train_op],feed_dict={x:batch_x,y:batch_y})
        if (i+1) % 500 == 0:
            print ('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i+1, loss_val,acc_val))
        if(i+1) % 5000 == 0:
            all_test_acc_val = []
            for j in range(test_steps):
                test_x,test_y = mnist.test.next_batch(batch_size)
                acc_val = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
                all_test_acc_val.append(acc_val)
            test_acc = np.mean(all_test_acc_val)
            print ('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc))

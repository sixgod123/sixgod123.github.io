from __future__ import print_function, division
# 必须在开头
import math
import datetime
starttime = datetime.datetime.now()
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import scipy.io as sio


class mydtw(object):
    def __init__(self, pflag=0):
        self.d = np.array([])

    def dtw(r, t):
        # 这是一个序列一个序列相比较
        (M, col) = np.shape(r)
        (N, col) = np.shape(t)
        d = np.zeros([M, N])
        # print("np.shape(d)",np.shape(d))   # (125,125)
        for i in range(M):
            for j in range(N):
                d[i, j] = np.sqrt(np.sum(np.square(r[i, :] - t[j, :])))
        D = np.zeros(np.shape(d))
        D[1, 1] = d[1, 1]
        for m in range(2, M):
            D[m, 1] = d[m, 1] + D[m - 1, 1]
        for n in range(2, N):
            D[1, n] = d[1, n] + D[1, n - 1]
        for m in range(2, M):
            for n in range(2, N):
                if min(D[m - 1, n], min(D[m - 1, n - 1], D[m, n - 1])) == D[m - 1, n - 1]:
                    D[m, n] = 2 * d[m, n] + min(D[m - 1, n], min(D[m - 1, n - 1], D[m, n - 1]))
                else:
                    D[m, n] = d[m, n] + min(D[m - 1, n], min(D[m - 1, n - 1], D[m, n - 1]))

        Dist = D[M - 1, N - 1]
        return Dist

    def compare(samples, Xp):
        print("np.shape(samples)", np.shape(samples))  # mp.shape(samples) (629,125,256)
        print("np.shape(Xp)", np.shape(Xp))  # np.shape(Xp) (32,125,256)
        # dist_min = np.zeros([np.shape(samples)[0], 1])
        dist_min = []
        dist_p = np.zeros([np.shape(samples)[0], np.shape(Xp)[0]])
        all_index = np.zeros([np.shape(samples)[0]])
        for i in range(np.shape(samples)[0]):
            sam = samples[i]
            # 与无标签例子按个比较
            for j in range(np.shape(Xp)[0]):
                ple = samples[j]
                dist_p[i, j] = mydtw.dtw(ple, sam)
            # 每个例子离得最近的正例距离是多少
            dist_min[i] = np.min(dist_p[i,:])
            # 或者说算距离之平均
            # dist_avg = np.sum(dist_p,axis = 1)


        # 全部无标签样本完事之后，一起做处理。
        h = np.where(all_index == 1)
        print("np.shape(h)", np.shape(h))
        # 将离正例更近的样本添加到正例，从无边前样本中删除
        Xp_new = Xp.append(samples[h])
        print("np.shape(Xp_new)", np.shape(Xp_new))
        samples_new = np.delete(samples[h])
        print("np.shape(samples_new)", np.shape(samples_new))
        return Xp_new, samples_new

class myRNN(object):
    def __init__(self,samples,Xp):
        ###### 设置参数 ######
        learning_rate = 0.01
        training_iters = 1000
        batch_size = 10
        display_step = 20  # 不知道是什么？
        seq_max_len = 125  # n_steps
        n_inputs = 256
        n_hidden = 128
        n_classes = 2

        ######## 生成数据 #######
        # 将array转换为列表
        samples = samples.tolist()
        Xp = Xp.tolist()
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(np.shape(Xp)[0]):
            self.Xlabels.append([1])
        for j in range(np.shape(samples)[0]):
            self.Xlabels.append([0])
        self.data = self.data.append(Xp)
        self.data = self.data.append(samples)
        n = len(self.data)
        list = []
        for i in range(0, n):
            list.append(i)
        random.shuffle(list)
        # 以上打乱顺序了

        a = []
        b = []
        for i in list:
            a.append(self.data[i])
            b.append(self.labels[i])
        if n_train == 500:
            self.data = a[0:500]
            # print('self.data',self.data)
            self.labels = b[0:500]
            print(self.labels)
        elif n_test == 210:
            self.data = a[500:710]
            # print('self.data',self.data)
            self.labels = b[500:710]
            print(self.labels)
        else:
            pass
    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        # print('batch_data----', batch_data)
        # print('batch_data.size----', np.shape(batch_data))
        return batch_data, batch_labels, batch_seqlen

    def dynamicRNN(x, seqlen, weights, biases):
        x = tf.transpose(x, [1, 0, 2])  # 循环神经网络要求输入要把循环维度放在第一位，比如对于一个list=[1,2,3,4,5,6,7,8]，循环次数是8
        x = tf.reshape(x, [-1, n_inputs])
        x = tf.split(axis=0, num_or_size_splits=seq_max_len,
                         value=x)  # tf.shape(x)此时为【(axis = 0的len)/seq_max_len,:】
            # 一个例子： 'value' is a tensor with shape [5, 30]
            # Split 'value' into 3 tensors along dimension 1
            # split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
            # tf.shape(split0)  # [5, 10]
            ## print('x.split----', x)
            ## print('x.split.type----', type(x))
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=_init_state, time_major=False)
        ## print('rnn_outputs----', outputs)    # tf.contrib.rnn.static_rnn函数的返回值是n_steps个【batch_size,n_hidden】形状的张量
        ## print('rnn_outputs.size----', np.shape(outputs))
        outputs = tf.stack(outputs)  # tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数
        ## print('outputs_stack----', outputs)  # Tensor("stack:0", shape=(20, ?, 64), dtype=float32)
        ## print('outputs_stack.size----', np.shape(outputs))
        outputs = tf.transpose(outputs, [1, 0, 2])
        # print(outputs)                         # Tensor("transpose_1:0", shape=(?, 20, 64), dtype=float32)
        batch_size = tf.shape(outputs)[0]
        ## print(batch_size)  # 128               # Tensor("strided_slice:0", shape=(), dtype=int32)
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)  # ?
        ## print('index----', index)
        ## print('index_size----', np.shape(index))
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # reshape成为[batch_size*n_steps,n_hidden]。
        ## print('outputs----', outputs)         # Tensor("Gather:0", shape=(?, 64), dtype=float32)
        ## print('outputs_size----', np.shape(outputs))
        # 可以看出index和gather操作是为了得到这一批数据中，每个list在最后一次有效循环（list长度）结束时的输出值。
        return tf.matmul(outputs, weights['out']) + biases['out']

    def main(self):
        # trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
        # testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
        trainset = mydata(n_samples=500)
        print('trainset.data.size', np.shape(trainset.data))
        # print(type(trainset))
        # print(trainset.next(batch_size))
        testset = mydata(n_samples=210)
        print('testset.data.size', np.shape(testset.data))
        # print('testset ',testset)
        # 设置占位符及隐藏层到输出层的参数
        x = tf.placeholder("float", [None, seq_max_len, n_inputs])  # 对于一批数据，x的三个维度分别表示[batch_size,n_steps,n_input]
        y = tf.placeholder("float", [None, n_classes])
        seqlen = tf.placeholder(tf.int32, [None])

####### 设置占位符及rnn隐藏层到输出层的参数 ########
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        ######### 训练和测试 ######
        pred = dynamicRNN(x, seqlen, weights, biases)
        print('pred----', pred)
        print('pred.size----', np.shape(pred))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # tf.argmax就是返回最大的那个数值所在的下标
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 计算一个张量各个维度的均值
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            while step * batch_size < training_iters:
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                # print('batch_x---', batch_x)
                # print('shape(batch_x)---', np.shape(batch_x))
                # print('batch_y---', batch_y)
                # print('shape(batch_y)---', np.shape(batch_y))
                # print('batch_seqlen---', batch_seqlen)
                # print('shape(batch_seqlen)---', np.shape(batch_seqlen))
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")
            test_data = testset.data
            # print('testdata ',test_data)
            print('test.data.size', np.shape(test_data))
            # test_data = tf.split(axis=0, num_or_size_splits=seq_max_len, value=test_data)
            # print('split_testdata ', test_data)
            # test_data = tf.stack(test_data)
            # print('stack_testdata ',test_data)
            test_label = testset.labels
            print('test_label', test_label)
            print('test.label.size', np.shape(test_label))
            test_seqlen = testset.seqlen
            print('test_seqlen', test_seqlen)
            print('test_seqlen.size', np.shape(test_seqlen))
            pred = sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})
            print("Testing Accuracy:",pred )
        return pred

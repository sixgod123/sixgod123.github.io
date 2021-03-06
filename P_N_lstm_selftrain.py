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
import heapq
# 调用堆排序算法模块
# lst = [12, 13, 1, 8, 10]
# min_n = 2
# temp = map(lst.index, heapq.nsmallest(min_n, lst))
# temp = list(temp)
# print(temp)

class mydtw(object):
    def __init__(self, pflag=0):
        self.d = np.array([])
    def dtw(r, t):
        # 这是一个序列一个序列相比较
        (M, col) = np.shape(r)
        # print("np.shape(r)",np.shape(r))
        # print("(M,col)",M, col)
        (N, col) = np.shape(t)
        # print("np.shape(t)", np.shape(t))
        # print("(N,col)",N, col)
        d = np.zeros([M, N])
        # print("np.shape(d)",np.shape(d))   # (125,125)
        for i in range(M):
            for j in range(N):
                # print("type(r)",type(r))
                # print("np.shape(r)",np.shape(r))    # （125,256）
                # print("r[i][:]",r[i][:])              # TypeError: list indices must be integers or slices, not tuple
                # print("np.shape(r[i][:])",np.shape(r[i][:]))
                # d[i, j] = np.sqrt(np.sum(np.square(r[i][:] - t[j][:])))  # TypeError: list indices must be integers or slices, not tuple
                # TypeError: unsupported operand type(s) for -: 'list' and 'list' 就是列表与列表没办法相减
                d[i, j] = np.sqrt(np.sum(np.square(np.array(r[i][:]) - np.array(t[j][:]))))

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
    def compare(samples, Xp, Xn):
        print("np.shape(samples)", np.shape(samples))  # np.shape(samples) (100,125,256)
        print("np.shape(Xp)", np.shape(Xp))            # np.shape(Xp) (10,125,256)
        # dist_min = np.zeros([np.shape(samples)[0], 1])
        dist_p = np.zeros([np.shape(samples)[0], np.shape(Xp)[0]])
        # dist_min = np.zeros([np.shape(samples)[0],1]) #  此时创建的为数组，数组没有index方法
        # dist_min = [[0]*np.shape(samples)[0]]    # 【【0,0，0,0,0 0 0 0 0 0 0】】
        # 加入我想要的是【【0】，【0】，【0】，。。。】 下面来实现
        dist_min = []
        for i in range(np.shape(samples)[0]):
            dist_min.append([0])                        # 注意加括号和不加括号是不一样的
        all_index = np.zeros([np.shape(samples)[0]])
        for i in range(np.shape(samples)[0]):
            sam = samples[i]
            # 与无标签例子按个比较
            for j in range(np.shape(Xp)[0]):
                ple = Xp[j]
                dist_p[i, j] = mydtw.dtw(ple, sam)
            # 每个例子离得最近的正例距离是多少
            dist_min[i] = np.min(dist_p[i,:])
            # 或者说算距离之平均
            # dist_avg = np.sum(dist_p,axis = 1)
        #  lst = [12, 13, 1, 8, 10]
        #  最近的5%作为正例
        # min_n = np.floor(0.05*np.shape(samples)[0])
        #  TypeError: 'numpy.float64' object cannot be interpreted as an integer
        # 选取最近的5%
        min_n = int(np.floor(0.08*np.shape(samples)[0]))
        print('min_n',min_n)
        small = map(dist_min.index, heapq.nsmallest(min_n, dist_min))
        small = list(small)
        # print("np.shape(Xp_new)", np.shape(Xp))

        #  最远的10%作为负例
        max_n = int(np.floor(0.15 * np.shape(samples)[0]))
        large = map(dist_min.index, heapq.nlargest(max_n, dist_min))
        large = list(large)
        return small,large
# 这个generate 函数需要用到sam_yhat
def generate(samples, Xp, Xn,small,large,Accurancy,sam_yhat,i,fir):
    # Xp_new = np.vstack((Xp, samples[small]))
    # Xp_new = Xp.append(samples[small])  # 1.Xp不是列表2.列表的索引不能是列表
    ###### 由于提前转换了 ####
    # Xp = Xp.tolist()
    if fir == 0:
        if 0.7 < Accurancy < 0.85:
            w1 = 0.5
            w2= 1 - w1
        elif Accurancy > 0.85 or i > 5:
            w1 = i/(i+2) * (Accurancy - 0.5) + 0.5
            w2 = 1 - w1
        else:
            w1 = Accurancy-0.5
            w2 = 1 - w1
        # sam_pred = tf.argmax(sam_yhat, 1)   # w1
        print(type(sam_yhat))               #
        print(np.shape(sam_yhat))           # (77,)
        dele = []
        for i in small:
            if w1* sam_yhat[i] + w2*0 <= 0.5:    # TypeError: Expected int64 passed to parameter 'y' of op 'LessEqual', got 0.5 of type 'float' instead.
                Xp.append(samples[i])            # TypeError: Expected int64, got 0.5 of type 'float' instead.
                dele.append(i)                    # During handling of the above exception, another exception occurred:
            else:
                pass
        for j in large:
            if w1 * sam_yhat[i] + w2*1 > 0.5:
                Xn.append(samples[j])
                dele.append(j)
            else:
                pass
    else:
        dele = []
        for i in small:
            Xp.append(samples[i])
            dele.append(i)
        # Xp_new = np.vstack((Xp, samples[small]))
        # print("np.shape(Xp_new)", np.shape(Xp))
        #  最远的10%作为负例
        # Xn_new = np.vstack((Xn, samples[small]))
        for j in large:
            Xn.append(samples[j])
            dele.append(j)
    # Xn_new = np.vstack((Xn, samples[small]))
    print("np.shape(Xn_new)", np.shape(Xn))
    samples_new = np.delete(samples, dele, axis=0)
    print("np.shape(samples_new)", np.shape(samples_new))
    return samples_new, Xp, Xn

###### 设置RNN参数 ######
learning_rate = 0.01
training_iters = 200
batch_size = 10
display_step = 10  # 不知道是什么？
seq_max_len = 125  # n_steps
n_inputs = 256
n_hidden = 64
n_classes = 2

class mydata(object):
    def __init__(self,samples,Xp,Xn,test_Xp,test_Xn,Ptrain,Ptest,Psample):
        ######## 生成数据 #######
        # 将array转换为列表
        # Xp = Xp.tolist()
        # Xn = Xn.tolist()
        self.train = []
        self.labels = []
        self.test = []
        self.seqlen = []    # 不知道是啥
        self.data = []
        self.batch_id = 0
        if Ptrain == 1:
            for i in range(np.shape(Xp)[0]):
                self.labels.append([1,0])
                self.seqlen.append(125)
                self.data.append(Xp[i])     # 千万不要用Xp = Xp.append(a)
            for j in range(np.shape(Xn)[0]):
                self.labels.append([0,1])
                self.seqlen.append(125)
                self.data.append(Xn[j])
            print("shape(self.labels)",np.shape(self.labels))
        # self.data.append(Xp)
        # self.data.append(Xn)          # 这样只能增加一个
            n = len(self.data)
        # 想要随机打乱顺序，这样来训练
            list = []
            for i in range(0, n):
                list.append(i)
            random.shuffle(list)
        # 以上打乱顺序了
            a = []
            b = []
            print("np.shape(data)[0]",np.shape(self.data)[0])   # 2  ????,难道append只能一个一个加吗
            # n_train = int(np.floor(0.75*np.shape(self.data)[0]))  # 3/4 的数据用于训练，1/4的数据用于测试
            # print("n_train",n_train)
            # n_test = np.shape(self.data)[0]-n_train
            for i in list:
                a.append(self.data[i])
                b.append(self.labels[i])
            print("np.shape(b)",np.shape(b))
            # self.data = a[0:n_train]
            self.data = a
            print('shape(self.data)_train', np.shape(self.data))
            # print('self.data',self.data)
            # self.labels = b[0:n_train]
            self.labels = b
            print('shape(self.data)_train', np.shape(self.labels))
            # print(self.labels)
            # self.seqlen = self.seqlen[0:n_train]
            # self.seqlen = self.seqlen
        elif Ptest == 1:
            for i in range(np.shape(test_Xp)[0]):
                self.labels.append([1,0])
                self.seqlen.append(125)
                self.data.append(test_Xp[i])     # 千万不要用Xp = Xp.append(a)
            for j in range(np.shape(test_Xn)[0]):
                self.labels.append([0,1])
                self.seqlen.append(125)
                self.data.append(test_Xn[j])
            # self.data = a[n_train:np.shape(self.data)[0]+1]   # 不加self的话：IndexError: tuple index out of range
            # print('shape(self.data)_test',np.shape(self.data))
            # self.labels = b[n_train:n_train+n_test+1]
            # print('shape(self.data)_test', np.shape(self.labels))
            # self.seqlen = self.seqlen[n_train:n_train+n_test+1]
        elif Psample == 1:
            self.data = samples
            for j in range(np.shape(samples)[0]):
                self.seqlen.append(125)
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

#### 运行程序 ####
# 未加标签
matfn = 'samples.mat'
data = sio.loadmat(matfn)
samples = data['samples']
##### 只用前100个samples #####
# a = list(range(100))
# samples = samples[a]
print(type(samples))      # (class 'numpy.ndarray')
# 读取近来的mat文件为<class 'numpy.ndarray'>
print(np.shape(samples))  # (100, 125, 256)

# 正例Xp
matfn = 'Xp.mat'
data = sio.loadmat(matfn)
Xp = data['Xp']
m = np.shape(Xp)[0]
n = int(np.floor(m/3))
Xp = Xp[range(n)]
print(type(Xp))        # (class 'numpy.ndarray')
# 读取近来的mat文件为<class 'numpy.ndarray'>
print(np.shape(Xp))    # (10, 125, 256)
# precision = np.array([])

# 用作测试的正例test_Xp
matfn = 'Xp.mat'
data = sio.loadmat(matfn)
test_Xp = data['Xp']
test_Xp = test_Xp[n:m]
print(type(test_Xp))        # (class 'numpy.ndarray')
# 读取近来的mat文件为<class 'numpy.ndarray'>
print(np.shape(test_Xp))    # (22, 125, 256)
# precision = np.array([])

# 用作测试的负例
matfn = 'Xn.mat'
data = sio.loadmat(matfn)
test_Xn = data['Xn']
print(type(test_Xn))        # (class 'numpy.ndarray')
# 读取近来的mat文件为<class 'numpy.ndarray'>
print(np.shape(test_Xn))    # (38, 125, 256)
# precision = np.array([])

def placehold(samples_new, Xp_new, Xn_new,test_Xp,test_Xn):
    # 正负类
    # trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
    # testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
    trainset = mydata(samples_new,Xp_new, Xn_new,test_Xp, test_Xn, Ptrain = 1,Ptest = 0,Psample = 0)
# 这里的trainset即相当于this
# print('trainset ',trainset)
# print(type(trainset))
# print(trainset.next(batch_size))
    testset = mydata(samples_new,Xp_new, Xn_new, test_Xp,test_Xn, Ptrain = 0,Ptest = 1,Psample = 0)
    samset = mydata(samples_new,Xp_new, Xn_new,test_Xp,test_Xn, Ptrain = 0,Ptest = 0,Psample = 1)

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
    return trainset,testset,samset,x,y,seqlen,weights,biases

def dynamicRNN(x, seqlen, weights, biases,re = 1):
    x = tf.transpose(x, [1, 0, 2])  # 循环神经网络要求输入要把循环维度放在第一位，比如对于一个list=[1,2,3,4,5,6,7,8]，循环次数是8
    print("np.shape(x)",np.shape(x))
    x = tf.reshape(x, [-1, n_inputs])
    print("np.shape(x)",np.shape(x))
    x = tf.split(axis=0, num_or_size_splits=seq_max_len,value=x)  # tf.shape(x)此时为【(axis = 0的len)/seq_max_len,:】
    print("np.shape(x)",np.shape(x))              # (125,)
            # 一个例子： 'value' is a tensor with shape [5, 30]
            # Split 'value' into 3 tensors along dimension 1
            # split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
            # tf.shape(split0)  # [5, 10]
            ## print('x.split----', x)
            ## print('x.split.type----', type(x))
    # tf.reset_default_graph()                    # 需要添加这一行来重置graph
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,reuse = True)
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    if re == 1:
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, reuse=True)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True,reuse = True)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forgrt_bias = 1.0,state_is_tuple = True,reuse = True)
    else:
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forgrt_bias = 1.0,state_is_tuple = True)
    # dropout防止过拟合
    dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    batch_size= 10
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # outputs, states = tf.contrib.rnn.static_rnn(dropout_lstm, x, dtype=tf.float32, sequence_length=seqlen)  # 出了问题
    outputs, states = tf.nn.dynamic_rnn(dropout_lstm, x, initial_state=_init_state, time_major=False)
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
    # i可以看出index和gather操作是为了得到这一批数据中，每个list在最后一次有效循环（list长度）结束时的输出值。
    return tf.matmul(outputs, weights['out']) + biases['out']
    # return tf.add(tf.cast(tf.matmul(outputs, weights['out']),tf.float32) ,tf.cast(biases['out'],tf.float32))

######### 训练和测试 ########
def RNN(object,trainset,testset,samset,x,y,seqlen,weights,biases,re = 1):
    if re == 1:
        pred = dynamicRNN(x, seqlen, weights, biases,re = 1)
    else:
        pred = dynamicRNN(x, seqlen, weights, biases,re = 0)

    print('pred----', pred)                  #  Tensor("add_1:0", shape=(?, 2), dtype=float32)
    print('pred.size----', np.shape(pred))  #  (?, 2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # tf.argmax就是返回最大的那个数值所在的下标
    print(tf.shape(correct_pred))   #Tensor("Shape_4:0", shape=(1,), dtype=int32)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 计算一个张量各个维度的均值
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # print('batch_x---', batch_x)
            print('shape(batch_x)---', np.shape(batch_x))
            print('batch_y---', batch_y)
            print('shape(batch_y)---', np.shape(batch_y))
            print('batch_seqlen---', batch_seqlen)
            print('shape(batch_seqlen)---', np.shape(batch_seqlen))
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
        Accurancy = sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})
        print("Testing Accuracy:",Accurancy )
        sam_data = samset.data
        sam_seqlen = samset.seqlen
        sam_pred = sess.run(pred,feed_dict = {x:sam_data,seqlen:sam_seqlen})
        print("np.shape(sam_pred)",np.shape(sam_pred))      # (77,2)
        print("type(sam_pred)",type(sam_pred))              # <class "numpy.ndarray">
        sam_max = tf.argmax(sam_pred, 1)
        print("np.shape(sam_max)", np.shape(sam_max))
        print("type(sam_max)", type(sam_max))               # <class 'tensorflow.python.framework.ops.Tensor'>
        sam_yhat = sess.run(sam_max)
        print("np.shape(sam_yhat)", np.shape(sam_yhat))
        print("type(sam_yhat)", type(sam_yhat))
    return Accurancy,sam_yhat

###### 最终运行程序 ######
accurancy = []
precision = 0
# 准确率
accurancy.append(precision)

# 负例
Xn= []          # Xn本来就是list
print("type(samples)",type(samples))
print("type(Xp)",type(Xp))
print("type(Xn)",type(Xn))
###### 提前转换Xp samples #######
Xp = Xp.tolist()
test_Xp = test_Xp.tolist()
samples = samples.tolist()
test_Xn = test_Xn.tolist()

# 还未训练lstm，此时sam_yhat不用去考虑
Accurancy = 0
sam_yhat = []
i = 0
small,large= mydtw.compare(samples,Xp,Xn)
[samples_new, Xp_new, Xn_new] = generate(samples, Xp, Xn,small,large,Accurancy,sam_yhat,i,fir = 1)
trainset,testset,samset,x,y,seqlen,weights,biases = placehold(samples_new, Xp_new, Xn_new,test_Xp,test_Xn)
Accurancy,sam_yhat = RNN(object,trainset,testset,samset,x,y,seqlen,weights,biases,re = 0)
accurancy.append(Accurancy)
# 两次准确率之差大于阈值
i = 1
while Accurancy - precision >= 0.01 or Accurancy < precision:
    # 再重新分一次
    i  = i + 1
    print("第%d次重复运行"%i)
    precision = Accurancy
    samples = samples_new
    Xp = Xp_new
    Xn = Xn_new
    print("type(samples)",type(samples))
    print("type(Xp)",type(Xp))
    print("type(Xn)",type(Xn))
    small, large = mydtw.compare(samples, Xp, Xn)
    [samples_new, Xp_new, Xn_new] = generate(samples, Xp, Xn, small, large, Accurancy,sam_yhat, i,fir=0,)
    trainset, testset,samset, x, y, seqlen, weights, biases = placehold(samples_new, Xp_new, Xn_new,test_Xp,test_Xn)
    # precision = pred
    # tf.reset_default_graph()  # ValueError: Tensor("rnn/Const:0", shape=(1,), dtype=int32) must be from the same graph as Tensor("ExpandDims:0", shape=(1,), dtype=int32).
    Accurancy, sam_yhat = RNN(object, trainset, testset,samset, x, y, seqlen, weights, biases,re = 1)
    accurancy.append(Accurancy)

x = list(range(i+1))
print(x)
print(np.shape(x))
y = accurancy
print(y)
print(np.shape(y))
plt.plot(x,y,'ob-')
plt.show()
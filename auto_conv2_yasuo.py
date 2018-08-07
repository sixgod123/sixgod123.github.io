"""
author=Aaron
python=3.5
keras=2.0.6
tensorflow=1.2.1
"""
############## 利用keras进行自编码 ###########
# 2018/1/22
# 刘晓丽
# 改版：对电熔镁炉图片进行压缩，利用keras框架进行特征的提取。
############################################
import datetime
starttime = datetime.datetime.now()
from keras import Input
import numpy as np
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model
# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import tensorflow as tf

########################

import glob as gb
# import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import xlwt


path = "E:\\photo_test"    #获取当前路径
count = 0
for root,dirs,files in os.walk(path):    #遍历统计
      for each in files:
             count += 1   #统计文件夹下文件个数
print (count)
image_path_head = "E:\\photo_test\\"
image_path_tail = ".jpg"

##########模型##########
# 定义encoder
input_img = Input(shape=(64, 128, 3))
print(input_img.shape)                  # (?, 64, 128, 3)

# x = MaxPooling2D(pool_size=(4,4),padding='same')(input_img)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)  # (?, 64, 128, 8)
x = Conv2D(8, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
print(x.shape)                          # (?, 32, 64, 8)
# x = MaxPooling2D((3, 3), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
print(x.shape)                          # (?, 16, 32, 16)
x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(24, (2, 2), strides = (2,2), activation='relu', padding='valid')(x) ####**************########
print(x.shape)                          # (?, 8, 16, 24)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
print(x.shape)                          # (?, 4, 8, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(32, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
print(encoded.shape)                          # (?, 2, 4, 32)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = Conv2D(16, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
# print(x.shape)                          # (?, 4, 16, 16)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# encoded = Conv2D(16, (2, 2), strides = (2,2), activation='relu', padding='valid')(x)
# print(encoded.shape)                          # (?, 2, 8, 16)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='valid')(x)
# print(encoded.shape)                        # (?, 4, 16, 32)
# encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
####平滑#####


# 定义decoder
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# print(x.shape)                         # ( ?, 2, 8, 32))
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# print(x.shape)                         # ( ?, 4, 18, 32)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# print(x.shape)                         # ( ?, 8, 32, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
print(x.shape)                         # ( ?, 4, 8, 32)
x = UpSampling2D((2, 2))(x)            ####****************####
x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
print(x.shape)                         # (?, 8, 16, 32)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
print(x.shape)                         # (?, 16, 32, 32)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)                         # (?, 32, 64, 32)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
print(decoded.shape)                   # (?, 64, 128, 3)
# decoded = UpSampling2D((4, 4))(x)
# print(decoded.shape)                   # (?, 256, 1024, 3)
#######################
#########训练###########
num_train = 36233
x_train = np.empty((num_train,64,128,3),dtype="float32")
for i in range(0,num_train):
    image_seq = i+1
    image_path = "%s%d%s" % (image_path_head, image_seq, image_path_tail)
    image = plt.imread(image_path).astype('float32') / 255. - 0.5
    # plt.imshow(image.reshape(256, 944, 3))
    x_train[i,:,:,:] = image
    #########训练数组图片的显示#########
    # plt.imshow(x_train[i,:,:,:].reshape(256, 944, 3))
    # plt.axis('off')
    # plt.show()
    ########plt.show()##########
    # 作用是一张一张的打印出图片，关掉一张，才会在此打印出第二张 #
print(x_train.shape)
# (10, 256, 944, 3)



savedata = []
# ########测试##########
for i in range(0,2):
    num_test = 36233
    x_test = np.empty((num_test, 64, 128, 3), dtype="float32")
    for j in range(0, num_test):
        # index =  j
        # 因为对于autoencoder而言，不用分测试集训练集，本身就是无监督算法
        image_seq = j +1
        image_path = "%s%d%s" % (image_path_head, image_seq, image_path_tail)
        image_test = plt.imread(image_path).astype('float32') / 255. - 0.5
        x_test[j, :, :, :] = image_test
        # img_test = cv2.imread(image_path)
        # print(img_test)
        # cv2.imshow('meilu_test', img_test)
        # cv2.waitKey(100)
        #########测试数组图片的显示#########
        # plt.imshow(x_test[i,:,:,:].reshape(256, 944, 3))
        # plt.axis('off')
        # plt.show()
        ########plt.show()##########
    print(x_test.shape)

    ####模型
    # 定义模型的输入
    auto_encoder = Model(input_img, decoded)
    # 定义模型的优化目标和损失函数
    # auto_encoder.compile(optimizer='sgd', loss='mean_squared_error')
    auto_encoder.compile(optimizer='adam', loss='mean_squared_error')
    # 定义编码模型
    encoder = Model(inputs=input_img, outputs=encoded)

    #####训练
    auto_encoder.fit(x_train, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
    #####测试
    # decoded_imgs = auto_encoder.predict(x_test)  # 测试集合输入查看器去噪之后输出
    encoded_imgs = encoder.predict(x_test)
    # print(decoded_imgs.shape)
    # (200, 2, 4, 32)#
    print(encoded_imgs.shape)
    savedata1 = encoded_imgs.reshape(encoded_imgs.shape[0],encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3])
    # (200,256)
    print("******")
    print(savedata1.shape)
    # savedata = [savedata,savedata1]
    # savedata = np.array(savedata)
    # save = np.concatenate((savedata, savedata1))
    # print(save.shape)

#############保存数据#########

# workbook = xlwt.Workbook(encoding='utf-8',style_compression=0)
# booksheet = workbook.add_sheet('mysheet',cell_overwrite_ok=True)
# # sheet.write(savedata,'savedata')
# for i,row in enumerate(savedata1):
#     for j,col in enumerate(row):
#         booksheet.write(i,j,col)
# workbook.save('test.xls')
np.savetxt("test_256.txt",savedata1)

endtime = datetime.datetime.now()
print('cost time',(endtime - starttime).seconds,'seconds')
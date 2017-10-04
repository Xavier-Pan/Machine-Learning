# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:42:16 2017

@author: sy
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
import time
def decode_idx3_ubyte(idx3_ubyte_file): 
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """ 
    # 读取二进制数据 
    bin_data = open(idx3_ubyte_file, 'rb').read() 
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽 
    offset = 0 
    fmt_header = '>iiii' 
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset) 
    print( '魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols) )
    # 解析数据集 
    image_size = num_rows * num_cols 
    offset += struct.calcsize(fmt_header) 
    fmt_image = '>' + str(image_size) + 'B' 
    images = np.empty((num_images, num_rows, num_cols)) 
    for i in range(num_images): 
        if (i + 1) % 10000 == 0: 
            print( '已解析 %d' % (i + 1) + '张' )
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols)) 
        offset += struct.calcsize(fmt_image) 
            
    return images 

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """ # 读取二进制数据 
    bin_data = open(idx1_ubyte_file, 'rb').read() 
    # 解析文件头信息，依次为魔数和标签数 
    offset = 0 
    fmt_header = '>ii' 
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset) 
    print( '魔数:%d, 图片数量: %d张' % (magic_number, num_images) )
    # 解析数据集 
    offset += struct.calcsize(fmt_header) 
    fmt_image = '>B' 
    labels = np.empty(num_images) 
    for i in range(num_images): 
        if (i + 1) % 10000 == 0: 
            print( '已解析 %d' % (i + 1) + '张' )
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0] 
        offset += struct.calcsize(fmt_image) 
            
    return labels

#=== get p(c) =====================================================================
def initial_prob(tra_data,tra_label,class_prob,num_class):#calculus P(c)
    N = len(tra_label)
    if N == 0:
        print("N == 0:\n")
    t= np.array([1]*len(tra_label))# for count lenth
    for i in range(10):
        mask = tra_label == i
        N_i = len(t[mask])# N for class i        
        num_class[i] = N_i
        class_prob[i] = N_i/N

#=== get p(x|c=k) =====================================================================
def prob_x_given_c(tra_data,tra_label,num_class,x,c):#calculus P(x)
#x:image
#c:class(0~9)
#num_class:amount of each class    
    mask = tra_label == c
    data = tra_data[mask]    
    N_c = num_class[c]#number of class_c   
    img_shape = np.shape(data)
            
    p = 1.
    for i in range(img_shape[1]):
        for j in range(img_shape[2]):
            num_feature_appear = 0
            bins = [0]*32
            for k in range(img_shape[0]):
             #   if (int)(data[k][i][j]/8) == (int)(x[i][j]/8): # if they are in same bin
                   # feature += 1
                   bins[(int)(data[k][i][j]/8)] += 1
            num_feature_appear = bins[(int)(x[i][j]/8)]
            if num_feature_appear == 0:# find the minimum bins and avoid divide zero              
                min_ = 60000
                for item in bins:
                    if item> 0 and item < min_:
                        min_ = item
                num_feature_appear = min_
                
                
            if num_feature_appear==0:
                print("num_feature_appear == 0")
            p *= num_feature_appear/N_c
    
    return p

#=== get p(x) =====================================================================
def prob_x(tra_data,tra_label,class_prob,x):#calculus P(x)
#x:image
#c:class(0~9)
#class_prob[k]:P(c=k)
    p_x = 1
    for i in range(10):
        pxc = prob_x_given_c(tra_data,tra_label,num_class,x,i)
        p_x += pxc*class_prob[i]#calculus log{P(x|c=k)P(c=k)} to avoid underflow        
    return p_x

#=== calculus accurate =====================================================================
def accurate(predict_y,y):#calculus P(x)
#predic_y: predicted label
#y:true label
    predict_y = np.array(predict_y)
    acc = 0.
    N = len(y)
    result = predict_y == y
    for item in result:
        if item == True:
            acc += 1
    return acc/N
        
#========================================================================

#def predict(tra_data,tra_label):
class_prob = np.array([.0]*10)
num_class = np.array([.0]*10)
'''
tra_data = decode_idx3_ubyte("train-images.idx3-ubyte")
tra_label = decode_idx1_ubyte("train-labels.idx1-ubyte")
test_data = decode_idx3_ubyte("t10k-images.idx3-ubyte")
test_label = decode_idx1_ubyte("t10k-labels.idx1-ubyte")    
initial_prob(tra_data,tra_label,class_prob,num_class)#get P(c=k) for all k
'''
p_list = [0.]*10

t_start = time.time()
predict = []
for i in range(3,10):
    #px =  prob_x(tra_data,tra_label,class_prob,test_data[i])#P(x) cost: 66 sec.
    t_start = time.time()
    for j in range(10):        
        pxc = prob_x_given_c(tra_data,tra_label,num_class,test_data[i],j)#P(x|c)        
        log_pcx = np.log(pxc)+np.log(class_prob[j])-np.log(px)    # log{P(c|x)}
        p_list[j] = log_pcx
    t_end = time.time()                    
    predict.append(np.argmax(p_list))
    print("time:",t_end-t_start)   
    plt.imshow(test_data[i], cmap='gray')
    plt.show()
    print(p_list)    
    print("predict = {}".format(predict[-1]))
t_end = time.time()    
print("accurate:",accurate(predict,test_label[3:10]))

aa = np.array([0]*10) == np.array([1]*10)
aa
'''
for i in range(3):
    plt.imshow(test_data[i], cmap='gray')
    plt.show()
    print("%d \n"%(i+1))
'''    
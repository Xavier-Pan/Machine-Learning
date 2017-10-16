# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:42:16 2017

@author: sy
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
import time

#=== parsing data =============================================================
def decode_MNIST_image(file):
    binary_data = open(file, 'rb').read() 
    offset = 0 
    fmt_header = '>iiii' # '>' stand for big endien (e.g 0x01234567 => 以 01 23 45 67 為順序), i stand for int.   
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, binary_data, offset)#解析byte

    image_size = num_rows * num_cols #784
    offset += struct.calcsize(fmt_header) # fmt_header的大小 即4個int= 32 byte = 4個欄位
    fmt_image = '>' + str(image_size) + 'B' # B mean 1 byte ,so  784 byte
    images = np.empty((num_images, num_rows, num_cols)) #empty array .mean no initial
    for i in range(num_images): 
        if (i + 1) % 10000 == 0: 
            print( '已完成 %d' % (i + 1) + '張' )
        images[i] = np.array(struct.unpack_from(fmt_image, binary_data, offset)).reshape((num_rows, num_cols)) 
        offset += struct.calcsize(fmt_image)             
    return images 

def decode_MNIST_label(file):
    """
    解析label
    param file: 位置
    return: label
    """
    binary_data = open(file, 'rb').read()     
    offset = 0 
    fmt_header = '>ii' 
    magic_number, num_images = struct.unpack_from(fmt_header, binary_data, offset)     

    offset += struct.calcsize(fmt_header) 
    fmt_image = '>B' 
    labels = np.empty(num_images) 
    for i in range(num_images): 
        if (i + 1) % 10000 == 0: 
            print( '已完成 %d' % (i + 1) + '筆' )
        labels[i] = struct.unpack_from(fmt_image, binary_data, offset)[0] 
        offset += struct.calcsize(fmt_image) 
            
    return labels

#=== get p(c) =================================================================
def initial_prob(tra_data,tra_label,class_prob,num_class,bins):#calculus P(c)    
    
    N = len(tra_label)
    if N == 0:
        print("N == 0:\n")
    t= np.array([1]*len(tra_label))# for count lenth
    for i in range(10):
        mask = (tra_label == i)
        N_i = len(t[mask])# N for class i        
        num_class[i] = N_i
        class_prob[i] = N_i/N
        
    #initial bins
    for c in range(10):#class
      mask = tra_label == c
      data = tra_data[mask]# only include images which belong to class c    
      img_shape = np.shape(data)
      for i in range(img_shape[1]):
        for j in range(img_shape[2]):                                
            for k in range(img_shape[0]):                 
                   bins[i,j,c,(int)(data[k][i][j]/8)] += 1

#=== get p(x|c=k) =============================================================
def prob_x_given_c(tra_data,tra_label,num_class,bins,x,c):#calculus P(x)
    '''
    x:image
    c:class(0~9)
    num_class:amount of each class    
    '''
    mask = tra_label == c
    data = tra_data[mask]    
    N_c = num_class[c]#number of class_c   
    img_shape = np.shape(data)
            
    p = 1. #p = 0 for log version
    for i in range(img_shape[1]):
        for j in range(img_shape[2]):
            num_feature_appear = 0            
            num_feature_appear = bins[i,j,c,(int)(x[i][j]/8)]
            if num_feature_appear == 0:# find the minimum bins and avoid divide zero              
                min_ = 60000
                for item in bins[i,j,c]:
                    if item> 0 and item < min_:
                        min_ = item
                num_feature_appear = min_
                
            if num_feature_appear==0:
                print("num_feature_appear == 0")
            p *= num_feature_appear/N_c
            #p += num_feature_appear/N_c #for log version
    
    return p
#=== get p(x|c=k) =============================================================
def log_prob_x_given_c(tra_data,tra_label,num_class,bins,x,c):#calculus P(x)
    '''
    x:image
    c:class(0~9)
    num_class:amount of each class    
    '''
    mask = tra_label == c
    data = tra_data[mask]    
    N_c = num_class[c]#number of class_c   
    img_shape = np.shape(data)
            
    p = 0 #for log version
    for i in range(img_shape[1]):
        for j in range(img_shape[2]):                        
            num_feature_appear = bins[i,j,c,(int)(x[i][j]/8)]
            if num_feature_appear == 0:# find the minimum bins and avoid divide zero              
                min_ = 60000
                for item in bins[i,j,c]:
                    if item> 0 and item < min_:
                        min_ = item
                num_feature_appear = min_                
            #p *= num_feature_appear/N_c
            p += np.log(num_feature_appear/N_c) #for log version
    
    return p

#=== get p(x) =================================================================
def prob_x(tra_data,tra_label,num_class,class_prob,bins,x):#calculus P(x)
    '''
    x:image
    c:class(0~9)
    class_prob[k]:P(c=k)
    '''
    p_x = 1
    for i in range(10):
        pxc = prob_x_given_c(tra_data,tra_label,num_class,bins,x,i)
        p_x += pxc*class_prob[i]#calculus log{P(x|c=k)P(c=k)} to avoid underflow        
    return p_x

#=== calculus accurate ========================================================
def accurate(predict_y,y):#calculus P(x)
    '''
    predic_y: predicted label
    y:true label
    '''
    predict_y = np.array(predict_y)
    acc = 0.
    N = len(y)
    result = predict_y == y
    for item in result:
        if item == True:
            acc += 1
    return acc/N

#=== prediction ========================================================
def prediction(tra_data,tra_label,num_class,class_prob,bins,test_data):#calculus P(x)
#test_data: image you want predict
    px = 1.
    p_list = np.array([0.]*10)
    for c in range(10):        
        log_pxc = log_prob_x_given_c(tra_data,tra_label,num_class,bins,test_data,c)#P(x|y)
        #print("P(x|y)=",pxc)
        log_pcx = log_pxc + np.log(class_prob[c])-np.log(px)    # log{P(y|x)}
        p_list[c] = log_pcx
    
    px = 0.
    for proba in p_list: #p(x) = sum{p(x|y)*p(y)} over all y
        px += np.exp(proba) 
    log_px = np.log(px)    
    return np.argmax(p_list),p_list - log_px

#=== calculus gaussain pdf ========================================================
def gaussain(x,mean=0,variance=1):#calculus P(x)    
    return 1./np.sqrt(2*np.pi*variance)*np.exp(-.5*np.square(x-mean)/variance)
#from scipy.stats import norm
#norm.pdf(211,loc = 0,scale = np.sqrt(30))
#=== get p(x|c=k) by gaussian =============================================================
def make_gaussian_x_given_c(tra_data,tra_label,num_class,mean,variance):#calculus P(x)
    '''
    x:image
    c:class(0~9)
    num_class:amount of each class 
    mean:[28,28,10] to store each feature's mean in different class
    variance:[28,28,10]
    '''
    for c in range(10):
        mask = (tra_label ==  c )
        data = tra_data[mask]# all data in class c
        N_c = num_class[c]#number of class_c   
        img_shape = np.shape(data)
        for i in range(img_shape[1]):
            for j in range(img_shape[2]):
                for k in range(img_shape[0]):#kth image
                    mean[i,j,c] += data[k,i,j]
                mean[i,j,c] /= N_c
                for l in range(img_shape[0]):
                    variance[i,j,c] += (data[l,i,j]-mean[i,j,c])*(data[l,i,j]-mean[i,j,c])
                variance[i,j,c] /= (N_c)


#=== predict By Gaussian ========================================================
def predictionByGau(mean,variance,class_prob,test_data):#calculus P(x)
#test_data: image you want predict
    p_list = np.array([0.]*10)    # store log{p(x|c)*p(c)}
    hyperparameter = np.exp(-10)
    image_shape = np.shape(test_data)
    for c in range(10):        
        log_pxc = 0 #log{P(x|c)}
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if variance[i,j,c] != 0:                
                    pp = gaussain(test_data[i,j],mean[i,j,c],variance[i,j,c])
                   # if pp==0:
                   #     print("x = {5} mean[{0},{1},{2}] = {3} variance[{0},{1},{2}] = {4}".format(i,j,c,mean[i,j,c],variance[i,j,c],test_data[i,j]))
                else:# to ignore variance == 0 (# variance =0 means that all training data in this pixel have the same value)
                    pp = gaussain(test_data[i,j],mean[i,j,c],hyperparameter)
                '''    
                if mean[i,j,c] == test_data[i,j]:
                        pp = 1.
                    else:
                        pp = hyperparameter # equal to penalty             
                '''    
                if pp == 0:                    
                    pp = hyperparameter
                
                log_pxc +=  np.log(pp)
#===============                    
        log_pcx = log_pxc + np.log(class_prob[c])    #non-normalize log{P(c|x)}
        p_list[c] = log_pcx
    px = 0.
    # evaluate p(x)
    for i,proba in enumerate(p_list):#p(x) = {sum(p(x|c)p(c)) over all c}
        px += np.exp(proba) 
    
    log_px = np.max(p_list) #approx. 
    #p_list = p_list - np.log(px) #logp(c|x) = logp(x|c)+logp(c)
    return np.argmax(p_list),p_list -log_px
        
#==== initial   =============================================================
class_prob = np.array([.0]*10)
num_class = np.array([.0]*10)
bins = np.zeros((28,28,10,32))
mean = np.zeros((28,28,10))
variance  = np.zeros((28,28,10))
tra_data = decode_MNIST_image("train-images.idx3-ubyte")#read data
tra_label = decode_MNIST_label("train-labels.idx1-ubyte")#read data
test_data = decode_MNIST_image("t10k-images.idx3-ubyte")#read data
test_label = decode_MNIST_label("t10k-labels.idx1-ubyte")#read data    


initial_prob(tra_data,tra_label,class_prob,num_class,bins)#get P(c=k) for all k  cost:9x sec.
make_gaussian_x_given_c(tra_data,tra_label,num_class,mean,variance)# to initial gaussian according to training data  cost:110 sec.

#==== predict   =============================================================

import sys
if sys.argv[1] == 0:
    print("mode: discrete")
else:
    print("mode: continuous")

p_list = [0.]*10
predict = []
batch = 10
num_data = len(test_data)
shuffle = np.arange(num_data)
#shuffle = np.random.choice(np.arange(num_data), size=num_data, replace=False, p=None)

t_start = time.time()
for i in range(batch):    
    px = 1. #px =  prob_x(tra_data,tra_label,num_class,class_prob,test_data[i])#P(x) cost: 66 sec.
    '''
    for j in range(10):        
        pxc = prob_x_given_c(tra_data,tra_label,num_class,bins,test_data[shuffle[i]],j)#P(x|c)        
        log_pcx = np.log(pxc)+np.log(class_prob[j])-np.log(px)    # log{P(c|x)}
        p_list[j] = log_pcx
    
    '''
    if sys.argv[1] == 0:#discrete mode
        predic_y, p_list = prediction(tra_data,tra_label,num_class,class_prob,bins,test_data[shuffle[i]])    
    else:
        predic_y, p_list = predictionByGau(mean,variance,class_prob,test_data[i])
    predict.append(predic_y)
    print("predict = {}\n".format(predict[-1]))
    print(p_list)    
t_end = time.time()

print("accurate:",accurate(predict,test_label[:batch]))
print("time:",t_end-t_start)
#== online learning ===============================================================

#==== make data ===============================================================
def makeData(n=10,k = 10):
# n:number of data(line)
# k:max number of flip coin in each time
    data = []
    for i in range(n):
        num_flip = np.random.choice(k) + 1
        data.append(np.random.choice(2,num_flip))# make 01 string of lenth = num_flip and store in list
    
    with open('flip_coin.txt','w') as f:
        for str01 in data:
            f.write(str(str01)[1:-1])
            f.write('\n')

#==== read data ===============================================================
def readData():
    data = []    
    data_ = []
    with open('flip_coin.txt','r') as f:        
            data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i][:-1]
        
    for item in data:
        k = 0
        for c in item:
            if c == '1':
                k += int(c)
        n = int((len(item)+1)/2)
        data_.append((n,k))
    return data_
#==== calculus  pdf of beta and binomial ===============================================================
def combination(n,k):
    n,k=3,2
    table = np.zeros((n+1,n+1))
    table[0,1] = 1
    for i in range(1,n+1):
        table[i,0] = 1
        for j in range(1,i+1):
            table[i,j] = table[i-1,j] + table[i-1,j-1]
    return table[n,k]
    
def Beta_normal_term(a,b):
    return gamma(a)*gamma(b)/gamma(a+b)

def Beta_pdf(theta,a,b):
    return np.power(theta,a-1)*np.power(1-theta,b-1)/Beta_normal_term(a,b)

def binomial_pdf(k,n,theta):
    return combination(n,k)*np.power(theta,k)*np.power(1-theta,n-k)
    
def posterior(theta,n,k,a,b):
    return Beta_pdf(theta,k+a-1,n+b-k-1)

def showBetaPdf():
    x = np.linspace(0,1,100)
    y = np.array([Beta_pdf(theta,2,5) for theta in x ])
    y2 = np.array([Beta_pdf(theta,.5,.5) for theta in x ])
    y3 = np.array([Beta_pdf(theta,5,1) for theta in x ])
    plt.plot(x,y,x,y2,x,y3)
    plt.show()

def onlineLearning():
    data = readData()
    a,b = 5,2
    for n,k in data:
        theta = k/(n-1+k)  #by MLE
        print("binomial:theta={}  prior:a={},b={}  posterior:a={},b={}".format(theta,a,b,a+k-1,b+n-k-1))        
        a += k-1
        b += n-k-1            
#=================================================================
#showBetaPdf()
#onlineLearning()

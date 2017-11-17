# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:08:57 2017

@author: Pan
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
'''
1. Logistic regression 
 
    INPUT: n (number of data point, D), mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 (m: mean, v: variance)
 
    FUNCTION: 
        a. Generate n data point: D1= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, where x and y are independently sampled from N(mx1, vx1) and N(my1, vy1) 
            respectively. (use the Gaussian random number generator you did for homework 3.).
        b. Generate n data point: D2= {(x1, y1), (x2 ,y2), ..., (xn, yn) }, where x and y are independently sampled from N(mx2, vx2) and N(my2, vy2) 
            respectively. 
'''
#===== normal generator =======================================================
#exercise 1.a 1.b
#==============================================================================

def normal_gen(mean = 0,var = 1,size =1):
    '''
    using Marsaglia polar method
    sampling from gaussian pdf
    var:variance    
    '''
    s = 1
    while(s>=1):
        u = np.random.uniform(-1,1)
        v = np.random.uniform(-1,1)
        s = u**2 + v**2
    
    temp = np.sqrt(-2*np.log(s)/s) 
    x = u * temp    
    return x*np.sqrt(var) + mean

def generate_data(n,mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    
    D1 = []
    D2 = []    
    for i in range(n):
        D1.append(np.array((normal_gen(mean = mx1,var = vx1),normal_gen(mean = my1,var = vy1),1)))#generate point like [x,y,1] 1 for bias
        D2.append(np.array((normal_gen(mean = mx2,var = vx2),normal_gen(mean = my2,var = vy2),1)))
    return D1,D2

#===== logistic regression =======================================================
#exercise 1.c
#==============================================================================
'''
        c. Use Logistic regression to separate D1 and D2. You should implement both Newton's and steepest gradient descent method during optimization, 
            i.e., when the Hessian is singular, use steepest descent for instead. 
            You should come up with a reasonable rule to determine convergence. (a simple run out of the loop should be used as the ultimatum) 
     OUTPUT: The confusion matrix and the sensitivity and specificity of the logistic regression applied to the training data D.
'''

'''
to judge wether use inverse
'''
#==============================
#Linear algebra utility
#==============================det(nomalized_H[:2,:2])
def det(A):#A=  det(nomalized_H)
    '''
    by calculus det(A)    
    '''
    s = 0. 
    shape = np.shape(A)
    if shape[0] ==2 and  2 == shape [1]:#stop when A is 2x2
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
   
    for j in range(shape[1]):
        if j == 1 :
            s += A[0,j]*det(np.hstack((np.array([A[1:,0]]).transpose(),A[1:,(j+1):])))*(-1)**(j%2)
        elif j == shape[1]-2:
            s += A[0,j]*det(np.hstack((np.array(A[1:,:j]),np.array([A[1:,(j+1)]]).transpose())))*(-1)**(j%2)
        elif j == 0 :
            s += A[0,0]*det(A[1:,1:])*(-1)**(j%2)
        elif j == shape[1]-1:
            s += A[0,j]*det(A[1:,:j])*(-1)**(j%2)
        else:
            s += A[0,j]*det((np.hstack((A[1:,:j],A[1:,(j+1):]))))*(-1)**(j%2)
    '''
    if s < 10**(-10):
        s = 0.
    '''
    return s

#==== calculus LU decomposition ===================== 
def LUDecom(A):
    U = np.copy(A)
    m = A.shape[0] #number of row
    L = np.eye(m) #initial L   
    for i in range(m-1):             
        for j in range(i+1,m):
            factor = (U[j,i]/U[i,i])            
            U[j,:] = U[j,:] - factor*U[i,:] # replace row i to j
            L[:,i] = L[:,i] + factor*L[:,j] # replace col j to i            
    return L,U

#==== calculus inverse for L ===================== 
def invL(L_):
    L = np.copy(L_) 
    n = L.shape[0] #number of row
    LInver = np.eye(n)
    for i in range(n):           
        for j in range(i+1,n):
            factor = L[j,i]/L[i,i]     
            #print("j=[{}]factor:{}".format(j,factor))
            LInver[j,:] = LInver[j,:] - factor*LInver[i,:] # replace col j to i
            #print("L:\n",LInver)
        LInver[i,:] /= L[i,i] # sacle row i
    return LInver

#==== calculus inverse for L ===================== 
def invByLU(A):    
    L,U = LUDecom(A)   
    return  matrix_mul(transpose(invL(transpose(U))),invL(L))
'''
def det2(A):
   
   # by calculus det(A)    
    
    s = 0    
    shape = A.shape    
    if A.size == 1:
        return A
        
    if shape[0] ==2 and  2 == shape [1]:#stop when A is 2x2        
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
   
    for j in range(shape[0]):        
        s += A[0,j]*det2(np.delete(np.delete(A,0,0),j,1))*(-1)**(j%2)            
    return s
'''
def transpose(A):
    AT = np.zeros((np.shape(A)[1],np.shape(A)[0]))
    for i in range(np.shape(A)[1]):
        for j in range(np.shape(A)[0]):
            AT[i,j] = A[j,i]        
    return AT

def matrix_mul(A,B):    
    m = np.shape(A)[0]
    n = np.shape(B)[1]
    l = np.shape(B)[0]
    AB = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            for k in range(l):
                AB[i,j] += A[i,k]*B[k,j]
    return AB

def inv(A):
    #return np.linalg.inv(A)
    return invByLU(A)

def dot(x,y):       
    s = 0
    for i1,i2 in zip(x,y):        
        s += i1*i2
    return s

def sigmoid(x):    
        return 1/(1+np.exp(-x))
#===========================
def acc(y_hat,y):
    s = 0
    for i in range(len(y)):
        if y_hat[i]>.5 and y[i] == 1:
            s+= 1
        if y_hat[i]<=.5 and y[i] == 0:
            s+= 1
    return s/len(y)
    #return sum(list(y_hat>.5) and list(np.array(y) == 1))/len(y)

def cost_fun(sig_Xw,y):
    y = np.array(y)
    s = 0
    for i in range(y.size):
        s += y[i]*np.log(sig_Xw[i,0]+10**-20) + (1 - y[i])*np.log(1-sig_Xw[i,0]+10**-20)
    return -s/y.size#(-dot(y,(np.log(sig_Xw+10**-20))[:,0]) - dot((1-y),np.log(1-sig_Xw+10**-20)[:,0]))/y.size

#============================
#plot confusion matrix
#============================
def confuseMat(y_hat,y):#confuseMat(sig_Xw,y)
    predic = (y_hat <= .5)[:,0]# aa = (sig_Xw>.5)[:,0]
    true_y = (np.array(y) == 0) #bb = (np.array(y) == 1)
    TP,TN = 0,0
    for i in range(predic.size):
        if predic[i] == true_y[i]:
            if predic[i] == True:#sum(aa == bb and aa == True)
                TP += 1
            else:
                TN += 1    
    FP = sum(predic == True) - TP
    FN = sum(predic == False) - TN
    
    print("\npredict     |  0(紅)|    1(藍)")
    print("-----------------------------")
    print(" true    0  |{:7d}|{:7d}".format(TP,FN))
    print("         1  |{:7d}|{:7d}".format(FP,TN))
    print("sensityvity:",TP/(TP + FN))
    print("specificity:",TN/(TN + FP))
    
#============================
#plot decision boundary
#============================
def decision_bound(X,y,w):
    n = X.shape[0]
    x1_ = [X[i][0] for i in range(n) if y[i]== 0]
    y1_ = [X[i][1] for i in range(n) if y[i]== 0]
    x2_ = [X[i][0] for i in range(n) if y[i]== 1]   
    y2_ = [X[i][1] for i in range(n) if y[i]== 1]
   
    plt.plot(x1_,y1_,'or')
    plt.plot(x2_,y2_,'ob')
      
    xx = np.linspace(-3,10,50)
    yy = np.linspace(-3,8,50)
    for i in range(xx.size):
        for j in range(yy.size):
            if sigmoid(dot([xx[i],yy[j],1.],w)) >0.5:# label == 1,plot green
                plt.plot(xx[i],yy[j],'x',color = 'c')
            else:# label == 0 ,plot yellow
                plt.plot(xx[i],yy[j],'x',color = '#db7093')
                #plt.plot(xx[i],yy[j],'x',color = '#fa8072')
#============================

def logistic_regression(n=30,mx1=0, vx1=1, my1=0, vy1=1, mx2=4, vx2=3, my2=4, vy2=1):
    '''
    n=50
    mx1=0
    vx1=1
    my1=0
    vy1=1
    mx2=4
    vx2=4
    my2=-1
    vy2=1  
    '''  
    history = [np.zeros((3,1)), np.zeros((3,1))]
     #   threshold = .3    
    D1,D2 = generate_data(n,mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
    X = D1 + D2
    X = np.array(X)
    #Xt = transpose(X)#
    Xt = X.transpose()
    y = [0 for j in range(n)] + [1 for j in range(n)]
    w = transpose(np.array([np.random.rand(3)]))
    maxLoop = 70
    count = 1
    cost = [1 ,cost_fun(sigmoid(matrix_mul(X,w)),y)]
    learn_rate = 1
    do_hessian = False
    sig_Xw = sigmoid(matrix_mul(X,w))    
    
    D = diag((sig_Xw*(1-sig_Xw))[:,0])#diagnal matrix for calculus Hessian        
    H = matrix_mul(Xt,matrix_mul(D,X))#hessian matrix    
    #normalize H
    min_element = H.min()
    if min_element == 0:
        min_element = 1
    nomalized_H = H/min_element
    do_hessian = ( det(nomalized_H) != 0)       
        
        
    while(abs(cost[-1] - cost[-2]) > .001 and count < maxLoop):    
        sig_Xw = sigmoid(matrix_mul(X,w))
        print("acc:",acc(sig_Xw[:,0],y))
        print("[",count,"]cost:",cost_fun(sig_Xw,y))
        print("w",w)
        print("sig_Xw.min():",sig_Xw.min())
        grad = matrix_mul(Xt,sig_Xw - transpose(np.array([y])))
        D = diag((sig_Xw*(1-sig_Xw))[:,0])#diagnal matrix for calculus Hessian        
        H = matrix_mul(Xt,matrix_mul(D,X))#hessian matrix    
               
        #normalize H for calculus det(H)
        if do_hessian:           
            w = w - learn_rate*matrix_mul(np.linalg.inv(H),grad)    #update weight
            w=w/dot(w[:,0],w[:,0])#to avoid overflow
            print("== Hessian ==")
        else:
            w = w - learn_rate*grad
      
        history.append(w)            
        cost.append(cost_fun(sigmoid(matrix_mul(X,w)),y))
        count += 1        
    '''
    #show learning curve
    plt.plot(cost[1:],'o-',)
    plt.title('cost funciton')
    plt.show()
    '''
    #confuse matrix
    confuseMat( sigmoid(matrix_mul(X,w)),y)
    #decision boundary
    decision_bound(X,y,w)

def diag(v):
    D_ = np.zeros((v.size,v.size))
    for i_ in range(v.size):
        D[i_,i_] = v[i_]
    return D_

#================================================    
#2. EM algorithm
#================================================
'''
INPUT: MNIST training data and label sets.
 
FUNCTION: 
                a. Binning the gray level value into two bins. Treating all pixels as random variables following Bernoulli distributions. Note that each pixel follows a different Binomial distribution independent to others.
                b. Use EM algorithm to cluster each image into ten groups. You should come up with a reasonable rule to determine convergence. (a simple run out of the loop should be used as the ultimatum) 
 
OUTPUT: For each digit, output a confusion matrix and the sensitivity and specificity of the clustering applied to the training data.
'''
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
        images[i] = np.array(struct.unpack_from(fmt_image, binary_data, offset)).reshape((num_rows, num_cols)) 
        offset += struct.calcsize(fmt_image)             
    print( '已完成 {} decode'.format(file) )
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
        labels[i] = struct.unpack_from(fmt_image, binary_data, offset)[0] 
        offset += struct.calcsize(fmt_image) 
    print( '已完成 {} decode'.format(file) )        
    return labels

#===
def confuseMat_EM(y_pre,y,cluster):#confuseMat(sig_Xw,y)
    predic = y_pre[y == cluster]         
    TP = sum(predic == cluster)
    TN = sum(y_pre[y != cluster] != cluster)    
    FP = sum(y_pre == cluster) - TP
    FN = sum(y_pre != cluster) - TN
    
    print("\npredict",cluster,"   |  0    |   1   ")
    print("-----------------------------")
    print("  true    0  |{:7d}|{:7d}".format(TP,FN))
    print("          1  |{:7d}|{:7d}".format(FP,TN))
    print("sensityvity:",TP/(TP + FN))
    print("specificity:",TN/(TN + FP))
    return TP

#==============
    
def confuseMat_EM2(matrix,cluster):#confuseMat(sig_Xw,y)
    TP = int(matrix[cluster,cluster])
    TN = 0
    FP = int(sum(matrix[cluster,:]) - TP)
    FN = int(sum(matrix[:,cluster]) - TP)
    for i in range(10):
        if i != cluster:
            TN += sum(matrix[i])
    TN = int(TN - FN)
  
    print("\npredict",cluster,"   |  yes  |   no  ")
    print("-----------------------------")
    print("  true  yes  |{:7d}|{:7d}".format(TP,FN))
    print("         no  |{:7d}|{:7d}".format(FP,TN))
    print("sensityvity:",TP/(TP + FN))
    print("specificity:",TN/(TN + FP))


def argmax(list_):
    m = 0
    for i in range(len(list_)):
        if list_[m] < list_[i] :
            m = i
    return m            

def cluster(pre_y,y):
    print("# of group",end="")
    for i in range(10):
        print("|{:5d}".format(i),end="")
    print("|total  |predict")
    print("==================================================================================\
          =====")
    confusion = np.zeros((10,10))
    for k in range(10):
        temp = y[pre_y==k]
        cluster = [0]*10#
        for i in range(temp.size):
            cluster[int(temp[i])] += 1
        print("{:10d}".format(k),end="")            
        
        for j in range(10):
            print("{:6d}".format(cluster[j]),end="")            
        print("{:6d} {:6d}".format(sum(cluster),argmax(cluster)),end="")            
        confusion[argmax(cluster),:] += np.array(cluster)
        print("")
  
    print("==================================================================================\
          =====")
    print(" sensity.:",end="")
    for j in range(10):
        print("  {:4.2f}".format(confusion[j,j]/sum(confusion[:,j])),end="")            
   
    print("")
    return confusion
#cluster(y_predict,tra_label)               
#======================calculus expection ==========================
def EM_cost(Z,PI,U):
    J = 0
    for i in range(N):
        for j in range(K):
            temp = PI[j,0]
            for d in range(X.shape[1]):
                temp += X[i,d]*np.log(U[d,j]+10**-323)+(1-X[i,d])*np.log(1-U[d,j]+10**-323)
            J += Z[i,j]*temp
    return J

def crop(images,crop_size):
    diff = int((28 - crop_size)/2)
    n = images.shape[0]
    new_images = np.zeros((n,crop_size,crop_size))
    for i in range(n):
        new_images[i,:,:] = images[i,diff:-diff,diff:-diff]    
    return new_images
#a = np.array([[1,2,3,4],[3,4,5,6]])
#a[0] = np.array([1,1,1,1])
def show_image(image):    
    if image.max() > 1:
        image /= 255.
    plt.imshow(image)
    plt.show()
    
def resize(images,factor = 2):
    new_images = np.zeros((images.shape[0],int(images.shape[1]/factor),int(images.shape[2]/factor)))
    print(new_images.shape)
    for i in range(new_images.shape[0]):
        for j in range(new_images.shape[1]):
            for k in range(new_images.shape[2]):               
                    new_images[i,j,k] = images[i,j*factor,k*factor] 
    return new_images

#==============================================================================
tra_data = decode_MNIST_image("train-images.idx3-ubyte")#read data
tra_data = tra_data[:20000]
tra_label = decode_MNIST_label("train-labels.idx1-ubyte")#read data
tra_label = tra_label[:20000]
infinite_min = 10**(-323)
K = int(max(tra_label) + 1)
N = tra_label.size
crop_Size = 18
factor = 1
temp_X = resize(crop(tra_data,crop_size = crop_Size),factor)#tra_data = np.floor(tra_data / 128.)
#temp_X = crop(tra_data,crop_size = crop_Size) #tra_data = np.floor(tra_data / 128.)
#show_image(tra_data[21,:,:])#temp_X
#show_image(temp_X[21,:,:])#temp_X
X = np.reshape(temp_X,(N,int(crop_Size*crop_Size/factor/factor)))#tra_data.max()
#show_image(crop(tra_data,crop_size = crop_Size)[0,:,:])
for i in range(X.shape[0]):
    for j in range(X.shape[1]):      
        X[i,j] = (int)(X[i,j]/128)        
Xt = transpose(X)
One = np.ones((1,K))
one_N = np.ones((N,1))

PI = np.ones((K,1))/K
U = np.random.rand(X.shape[1],int(K))/2+.25#0.25 - .75
#nomalize uk
for i in range(U.shape[1]):
    temp = sum(U[:,i])     
    for j in range(U.shape[0]):
        U[j,i] /= temp    #U[:,0]
Z = np.zeros((N,K))
Pnk = np.zeros((N,K))

count = 1
y_predict = np.zeros((N,))
cost = [0,2]#initial ,for record the expectation function

record_correct = []
#while(abs(cost[-1] - cost[-2]) > .001):    
while(cost[-1]-cost[-2] >.1 ):    
    #Pnk = np.exp((matrix_mul(X,np.log(U)) + matrix_mul(1 -X,np.log(1 - U))))# P(Xn|uk) 600  8sec.
    time_s = time.time()  
    for i in range(N):
        for j in range(K):
            temp = 1
            for d in range(X.shape[1]):
                if X[i,d] > 0:#X.max()
                    temp *= U[d,j]
                else:
                    temp *= 1-U[d,j]                   
            Pnk[i,j] = temp    

    D = diag(PI[:,0])#for calculus P(xn|uk)*pi_k
    expenPI = np.matmul(PI,One)#[PI PI PI PI...] for Z
    #updata    
    Z = np.matmul(Pnk,D)/(np.matmul(Pnk,expenPI)+infinite_min)
    PI = np.matmul(Z.transpose(),one_N)/N   
    U = np.matmul(np.matmul(Xt,Z),diag(1/(N*PI[:,0])))  
    #=====
    '''
    for i in range(N):
        for j in range(K):
            #if (np.matmul(Pnk,PI)[j,0]+10**(-400)) ==0:
             #   print("(",i,",",j,")")
            Z[i,j] = Pnk[i,j]*PI[j,0]/(np.matmul(Pnk,PI)[i,0]+10**(-323))
    '''
    #====i=2,j=8
      
    '''
    temp = 0.
    for i in range(Z.shape[0]):#Z = [600*10]
        temp += Z.transpose()[:,i]
    PI = np.array([temp]).transpose()/N
    '''
    #====
            
    cost.append(EM_cost(Z,PI,U))    
    print("time:{:.3f}".format(time.time() - time_s))
    total_cor = 0
    for i in range(N):
        y_predict[i] = Z[i].argmax()
        if Z[i].max()>.99:
            total_cor += 1
            
    confu_matrix = cluster(y_predict,tra_label)
    print("[",count,"] confiden >.99的圖:",total_cor)
    count += 1
    
    for j in range(K):
        confuseMat_EM2(confu_matrix,j)
    #confu_matrix[9]
    
    '''
    temp = []
    temp2 = []
    for i in range(y_predict.size):
        if y_predict[i] == 5 and tra_label[i]== 4:
            temp.append(i)
        elif y_predict[i] == 4 and tra_label[i]== 4:
            temp2.append(i)
    for i in range(10):
        show_image(temp_X[temp[i],:,:])#temp_X
    print("++++++++++++++++++++++++++++++++++++++")
    for i in range(10):
        show_image(temp_X[temp2[i],:,:])#temp_X
        #show_image(np.floor(tra_data[temp2[i],:,:]+.5))#temp_X
    '''
    
#==count # of each calss====================

#plt.plot(record_correct)
'''
cc = [0]*10    
for i in tra_label:
    cc[int(i)] += 1
'''

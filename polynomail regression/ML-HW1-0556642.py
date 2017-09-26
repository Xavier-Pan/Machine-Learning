# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 23:23:37 2017

@author: sy
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import time


def makeTrainingData(n):        
    x = np.linspace(-5,10,n)# normalize to range -5~+5 
    #y = np.sin(x/n*(4*np.pi)) + np.random.normal(scale = 0.1,size = n)#add gaussian noise
    y = np.power(x,5)+ (-10)*np.power(x,4) + (10)*np.power(x,3) + (-4)*np.power(x,2)+ x + 15 + np.random.normal(scale = 0.1,size = n)#add gaussian noise
    with open('.\\tra_data2.txt','w') as f:
        for i in range(len(x)):
            f.write(str(x[i])+','+str(y[i]))
            f.write('\n')
#makeTrainingData(50)
#x,y = readData('tra_data2.txt')
#plt.plot(x,y)
def readData(path):
    x=[]
    y=[]
    data = []
    with open(path,'r') as f:            
            data = f.readlines()            
    for item in data:
        x.append(float(item.split(',')[0]))
        y.append(float(item.split(',')[1]))                
    return x,y

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
    return  matMul(transpose(invL(transpose(U))),invL(L))

#==== do matrix transpose =====================
def transpose(A):    
    m = A.shape[0]
    n = A.shape[1]
    transA = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            transA[i,j] = A[j,i]
    return transA

#==== do matrix multiply =====================
def matMul(A,B):    
    if A.shape[1] != B.shape[0]:
        print("matMul(A,B):the matrix dimension did not match!")
        return -1
        
    m = A.shape[0]
    l = A.shape[1]
    n = B.shape[1]
    C = np.zeros([m,n])    
    for i in range(m):
        for j in range(n):
            for k in range(l):
                C[i,j] += A[i,k]*B[k,j]
    return C
#aaa = matMul(transpose(np.array([[1,2,3]])),np.array([[1,2,3]]))
#=== square function ==========================
def LSE(y,y_hat):    
    s = 0.
    for i in range(len(y)):
        s +=  (y[i,0]-y_hat[i,0])*(y[i,0]-y_hat[i,0])
    return s/len(y)

#=== design matrix ==========================
def makeDisignMat(x,n):
    m = len(x)
    A = np.zeros([m,n+1])
    for j in range(n+1): #j:0,1,...,n
        A[:,j] = np.power(x,n-j)        
    return A

#=== solve Ly = b ==========================
def equ4L(L,b):
    y = np.copy(b)    
    #b = np.mat(b)
    Lt = np.copy(L)
    #np.hstack((Lt,b))#concate matrix L and vector b
    for i in range(Lt.shape[0]-1):
        for j in range(i+1,Lt.shape[0]):
            y[j] -= y[i]*Lt[j,i]        
    return y

#=== solve Ux = y ==========================
def equ4U(U,y):
    x = np.copy(y)    
    #b = np.mat(b)
    #Ut = np.copy(U)
    #np.hstack((Lt,b))#concate matrix L and vector b
    num_row = U.shape[0]
    for i in range(num_row-1):
        for j in range(i+1,num_row):
            k = num_row -1 - j# index k = n-2-i,...,0
            l = num_row -1 - i# index l = n-1 ,n-2 , ... ,1
            x[k] -= (U[k,l]/U[l,l])*x[l]

    for i in range(num_row):#equal to devide diagnal items to make it be 1
        x[i] /= U[i,i]        
    return x
#AA = np.array([[3,4,5],[0,5,1],[0,0,9]])
#yy = np.array([1.,2.,3.])
#a = np.array([equ4U(AA,yy)])
#matMul(AA,transpose(a))
#===find inverse by solve Ax=b ===================================
def invByEqu(A):    
    n = A.shape[0]
    iden = np.eye(n)
    invMat = np.zeros([n,n]) #initial inverse matrix 
    columList = []
    L,U = LUDecom(A)
        #== find inverse ===
    for i in range(n):
        t = equ4L(L,iden[i,:])
        xx = equ4U(U,t)
        columList.append(xx)
    
    for i in range(n):
        for j in range(n):
            invMat[i,j] = columList[j][i]
    return invMat
#invByEqu(np.eye(3))    
#=== plot real curve and predict curve ======================
def plotCurve(x,y,y_hat,err=0):
    cur1, = plt.plot(x,y_hat)
    cur2, = plt.plot(x,y)
    plt.legend(handles = [cur1,cur2],labels = ['predict','real'],loc = 'best')
    plt.title("Error:"+str(err))
    plt.show()
    
#=== solve inverse by LUx =b =====================================

    #===== initial =========
path = sys.argv[1]
n = int(sys.argv[2])
lamda = float(sys.argv[3])
x,y = readData(path)
x = np.array(x)
y = transpose(np.array([y]))
    #===== solve linear regression =========
iden = np.eye(n+1)#identity matrix
A = makeDisignMat(x,n)
Atrans = transpose(A)
AtA = matMul(transpose(A),A)
invMat = invByEqu(AtA+lamda*iden)    
#invMat = invByLU(AtA+lamda*iden)
x_hat = matMul(invMat,matMul(Atrans,y))
y_hat = matMul(A,x_hat)#predict y

    #==show equaltion===================================
error = LSE(y,y_hat)
s =''
for i,item in enumerate(x_hat):
    s+='+' + str(item[0]) + '*' + 'x' + str(n-i)
    s=s[1:]
print("eqution:",s)
print("error:",error)
    #===plot result ==========
plotCurve(x,y,y_hat,error)


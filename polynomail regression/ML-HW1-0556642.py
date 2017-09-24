# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 23:23:37 2017

@author: 順益
"""
import numpy as np

def findNonZeroRowIndex(A,i):
    return -1

def swap(B):
    C = B
    t = np.copy(C[0,:] )
    C[0,:] = C[1,:] 
    C[1,:] = t
    return C

#swap colum i and colum k
def swapCol(B,i,k):
    iCol = np.copy(B[:,i])
    B[:,i] = B[:,k]
    B[:,k] = iCol
        

#!!!!!(not complete)==== calculus LUP  which mean PA equal to LU ===================== 
def LUPDecom(A):#PA = LU for case that A did't have LU decomposition
    U = np.copy(A)
    m = A.shape[0] #number of row
    L = np.eye(m) #initial L
    P = np.eye(m) #initial P
    order = np.arange(m)# record order
    for i in range(m-1):
        if U[i,i] == 0:
            k = findNonZeroRowIndex(U,i) #find non-zero pivot from row i.it return -1 if failure
            if k >= 0:# find some index
                swap(U,i,k)
                swapCol(L,i,k) # equal to multiply a elementary matrix in right side
            else:
                continue #skip this 
       # U[i,:] /= U[i,i] # to make the pivot to be 1
        for j in range(i+1,m):
            factor = (U[j,i]/U[i,i])
            U[j,:] = U[j,:] - factor*U[i,:] # replace row i to j
            L[:,i] = L[:,i] + factor*L[:,j] # replace col j to i
        #print("[{}] U:{} ".format(i,U))
        #print("L:{} ".format(L))
    return L,U

#==== calculus LU decomposition ===================== 
def LUDecom(A):
    U = np.copy(A)
    m = A.shape[0] #number of row
    L = np.eye(m) #initial L
    for i in range(m-1):
        if U[i,i] == 0:
            k = findNonZeroRowIndex(U,i) #find non-zero pivot from row i.it return -1 if failure
            if k >= 0:# find some index
                swap(U,i,k)
                swapCol(L,i,k) # equal to multiply a elementary matrix in right side
            else:
                continue #skip this 
       # U[i,:] /= U[i,i] # to make the pivot to be 1
        for j in range(i+1,m):
            factor = (U[j,i]/U[i,i])
            U[j,:] = U[j,:] - factor*U[i,:] # replace row i to j
            L[:,i] = L[:,i] + factor*L[:,j] # replace col j to i
        #print("[{}] U:{} ".format(i,U))
        #print("L:{} ".format(L))
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

#==== do matrix transpose =====================
def transpose(A):    
    n = A.shape[0]
    transA = np.eye(n)
    for i in range(n):
        for j in range(n):
            transA[i,j] = A[j,i]

#=== square function ==========================
def LSE(y,y_hat):    
    return np.matmul(y-y_hat,y-y_hat)
#========test  =============================================    
A = np.array([[3.,-1.,2.],[6.,-1.,5.],[-9.,7.,3.]])
L,U = LUDecom(A)
L9 = invL(L)
U9 = np.transpose(invL(np.transpose(U))) # due to trans(inv(U)) ==  inv(trans(U))
result = np.matmul(U9,U)#for check result 
AInver = np.matmul(U9,L9)

L = np.array([[1,0,0],[2,1,0],[3,4,1]])

np.linalg.inv(np.transpose(U))# inverse
np.linalg(A)# inverse
print("L9*L=\n",L9*L)
print("L:\n",L)
print("U:\n",U)


#===read data & initial ===========
lamda = 0.001
n =  3
y = np.random.choice(range(100),size=5, replace=True)
size_data = y.shape[0]
x = np.arange(size_data)
A = np.ones([size_data,n+1])

for j in range(n+1):
    A[:,j] = np.power(x,n-j)
    np.power(x,2)
#np.power([1,2,3,4,5],2)
#========solve AtA + lambda*I ====================
Atrans = np.transpose(A)
AtA = np.matmul(Atrans,A)#transpose(A)*A
I = np.eye(A.shape[1])#initial
L,U = LUDecom(AtA + lamda*I)#LU decomposition
L_inv = invL(L) #inverse L
U_inv = np.transpose(invL(np.transpose(U))) # inverse U. due to trans(inv(U)) ==  inv(trans(U))
AtAInver = np.matmul(U_inv,L_inv)#evaluate A inverse
np.linalg.inv(AtA + lamda*I)# test inverse
x_hat = np.matmul(np.matmul(AtAInver,Atrans),y)
predict_y = np.matmul(A,x_hat)#evaluate predict result
print("error:{}".format(LSE(y,predict_y)))#lease square error
#printEq(x,n) #print equal
#=== plot graph ===========================
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()

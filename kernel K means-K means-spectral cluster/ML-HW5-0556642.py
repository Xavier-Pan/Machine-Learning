# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:12:44 2017

@author: pan
"""
import numpy as np
import os
import time
import matplotlib.pyplot as plt  
import matplotlib  
import copy
from matplotlib import animation

def initial():  
    co = list(matplotlib.colors.cnames.keys()  )    
    for i in range(len(co)):
        if (i+1)%4==0:
            color.append(co[i])            
        '''
        for i in range(2):
            plt.plot(np.random.rand(1),np.random.rand(1),'o',color = color[i])            
        ''' 
    
def kernel(x,y,v = .5):    
    z = x-y
    return np.exp(-z.dot(z)/(2*v))    
    #return np.tanh(5*sum(x*y) +3)

def load_data(fileName='test1_data.txt'):
    path = r"C:\Users\順益\Desktop\106-ML\HW5"
    path += "\\"+fileName
    data = []
    with open(path,'r') as f:
        data = f.readlines()
        
    #data = aa
    for i in range(len(data)):
        tup = data[i].strip('\n').split()
        data[i]=(float(tup[0]),float(tup[1]))
    
    t_data = np.zeros((len(data),2))
    for i in range(len(data)):
        t_data[i,:] = np.array([list(data[i])])#to transfer it to matrix form
        
    return t_data


#visualize(aa,np.zeros(len(aa)))    
def visualize(X,y,Mean,count,centroid=True):# at most 37 color y= np.zeros(len(aa)) X = aa
    num_class = int(max(y)) + 1
    for k in range(num_class):
        plt.plot(X[y==k,0],X[y==k,1],'o',color = color[k])
        if centroid:
            plt.plot(Mean[k,0],Mean[k,1],'*',color = color[k],markersize = 7)     
    plt.draw()
    plt.savefig(r"C:\Users\順益\Desktop\106-ML\HW5\img"+"\\"+str(count)+".jpg")
    plt.show()    
    print('save image '+str(count))

def k_means(X,count,k=2):#  X = np.random.rand(3,2)
    N = X.shape[0]
    Mean = np.random.rand(k,2)#N,k = 10,3
    selectedPoint = np.random.choice(N, size=k, replace=False)
    for i in range(k):
        Mean[i,:] = X[selectedPoint[i],:]
    y = np.random.randint(k,size= N)    
    hitory = copy.deepcopy (Mean)
    #count = 0
    while(1):
        #count for save image
        count+= 1
        #assign class     
        for i in range(N):
            mini = sum(np.square((X[i,:]-Mean[0,:])))
            label = 0                           
            for j in range(k):                
                temp = sum(np.square((X[i,:]-Mean[j,:])))
                if temp < mini:
                    mini = temp
                    label = j
            y[i] = label   
        #evaluate mean
        for i in range(k):#i=0
            indicator = (y==i)
            if sum(indicator)>1:#  np.array([sum(Mean[[0,0],:])]).shape
                Mean[i,:]= np.array([sum(X[indicator,:])])/sum(indicator)
            elif sum(indicator)== 1:#in case only one row,then sum() will return a scalar
                Mean[i,:]= X[indicator,:]
                #for j in range(len(Mean[i,:])):
                #    Mean[i,j]= X[indicator,j]
       
        visualize(X,y,Mean,count,False)# at most 37 color
        time.sleep(.3)
        
        if np.allclose(hitory , Mean):#weather jump out the while loop
            break
        else:
            hitory = copy.deepcopy (Mean)        
    return y

#==== kernel k-means=========
#
#============================   
def kernel_Kmeans(X,count,y,k=2,plot = True):#  X = np.random.rand(3,2)
    N = X.shape[0]
    Mean = np.zeros((k,2))#N,k = 10,3
  #  y = np.random.randint(k,size= N)    
    new_y = np.random.randint(k,size= N)    
    Ck = [0]*k  
    flat = 1
    while(flat):
        #t1 = time.time()
        flat = 0        
        count+= 1#count for save image
        #===evaluate cluster size                   
        for i in range(k):    
            Ck[i] = int(sum(y==i))#calculus size of cluster         
        #===calculus term3
        term3 = [0]*k#k=2        
        for  j in range(k):
            X_hat = X[y==j,:]
            for i in range(Ck[j]):                
                for jj in range(i+1,Ck[j]):
                    term3[j] += kernel(X_hat[jj,:],X_hat[i,:]) 
            term3[j] *= 2
            for i in range(Ck[j]):          
                term3[j] += kernel(X_hat[i,:],X_hat[i,:]) 
            term3[j] /= Ck[j]**2 #
        #===assign class     
        for i in range(N):#for all points                        
           # t1 = time.time()
            #Kii = kernel(X[i,:],X[i,:])      
            min_cost = float('inf')     
            x = X[i,:]                    
            for j in range(k):#for all clusters         
                #======               
                X_hat = X[y==j,:]                
                term = 0.              
                for ii in range(Ck[j]):
                    term += kernel(X_hat[ii,:],x)                   
                term *= 2/Ck[j]                            
                c = -term + term3[j]#first term dosen't matter
                #======
                if c < min_cost :
                    min_cost = c
                    new_y[i] = j                    
                   # flat = 1#mean change has happen 
        if sum(y!=new_y):
            flat = 1#not converge
        y = copy.deepcopy(new_y)        
        #plot image           
        # print("time:",time.time() - t1)
        if plot:
            visualize(X,y,Mean,count,False)# at most 37 color                 
    return y

#============================

from scipy import linalg
def spectral(X,y,k):    
    N = X.shape[0]
    W = np.zeros((N,N))   
    D = np.eye(N)
    I = np.eye(N)
    y = np.random.randint(k,size= N)  
    for i in range(N):
        s = 0
        for j in range(N):
            W[i,j] = kernel(X[i,:],X[j,:])
            s += W[i,j]
        D[i,i] = s
    #calculus laplacian
    L = D - W
    eigvals, eigvecs = linalg.eigh(L, I)
    
    U = eigvecs[:,:k]
    print(U.shape)
    #y= kernel_Kmeans(U,count,k=2,plot = False)
    y= k_means(U,count,2)
    visualize(X,y,np.zeros((len(y),2)),0,False)
    return U
#============================
#import os.getcwd()
#a = np.array([[12,22],[1,3]]).transpose()

color = []# for plot
#if __name__ =='__main__':
initial()

file_label = 'test2_ground.txt'
file = 'test2_data.txt'
data = load_data(fileName = file)
#== load true label
label = open(file_label,'r').readlines()
label = np.array([int(item.strip('\n')) for item in label])
#==  
visualize(data,np.ones(data.shape[0])  ,np.zeros((data.shape[0],2)),0,False)
count = 0
#y = k_means(data,count,2)
y = kernel_Kmeans(data,count,label)
#visualize(data,y,np.zeros((len(y),2)),0,False)
#U = spectral(data,y,2)#X =data




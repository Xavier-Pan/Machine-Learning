# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:11:10 2017

@author: sy
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import gamma
import time

#===== normal generator =======================================================
#exercise 1.a
#==============================================================================
'''
(1.a). univariate gaussian data generator 
          INPUT: expectation value (m), variance (s)
          OUTPUT: one outcome ~ N(m, s)
'''
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
#====== util ========================================================
def show_gaussian_sampling():
    x = [normal_gen(0,10) for i in range(30000)]
    plt.hist(x,70)
    plt.xlim(-30,30)
    plt.ylim(0,1400)
    plt.show()

def dot(x,y):       
    s = 0
    for i1,i2 in zip(x,y):        
        s += i1*i2
    return s

#== polynomial basis linear model =============================================
#exercise 1.b
#==============================================================================
'''    
(1.b).polynomial basis linear model (y = WTPhi(x)+e ; e ~ N(0, a)) data generator
          INPUT: basis number (n; ex. n=2 -> y = w0x0 +w1x1), a, w
          OUTPUT: y
          NOTE: there is an internal constraint: -10.0 < x < 10.0, x is uniformly distributed.
'''

def basis_lin_model(n,a,w):
    e = normal_gen(mean = 0,var = a)
    x = np.random.uniform(-10,10)
    phi_x = [x**i for i in range(n)]  
    y = dot(phi_x,w) + e
    return x,phi_x,y

#== sequencial estimate =======================================================
#exercise 3
#==============================================================================
'''
2. Sequential estimate the mean and variance from the data given from the univariate 
              gaussian data generator (1.a).
        NOTE: you should derive the recursive function of mean and variance 
                based on the sequential esitmation. 
        INPUT: m, s as in (1.a)
        FUNCTION: call (1.a) to get a new data point from N(m, s), use sequential 
                  estimation to find the current estimates to m and s.,repeat 
                  until the estimates converge.
        OUTPUT: print the new data point and the current estimiates of m and s 
                in each iteration.
'''
def sequen_estime(m,s):
    n = 1    
    sample_var = 0    
    sample_mean = normal_gen(mean = m,var = s)     
    print("[1]point:{}".format(sample_mean))
    sum_square = 0
    threshold = .01
    while abs(sample_mean - m) > threshold or abs(sample_var - s) > threshold:    
        n += 1
        new_point = normal_gen(mean = m,var = s)                
        update_mean = sample_mean + (new_point - sample_mean)/n# update    
        sum_square += (new_point - sample_mean )*(new_point - update_mean)    
        # update      
        sample_var = sum_square/(n-1)
        #sample_var = (n-2)/(n-1)*sample_var + np.square(new_point - sample_mean)/n
        sample_mean = update_mean 
        print("[{}]point:{:.3f} mean:{:.3f} variance:{:.3f}".format(n,new_point,sample_mean,sample_var))
        

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

def one_rank_matrix(x):
    n = len(x)
    xxT = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            xxT[i,j] = x[i]*x[j]
    return xxT
#=== Baysian Linear regression ================================================
'''
3. Baysian Linear regression
      INPUT: the precision (i.e., b) for initial prior w ~ N(0, b-1I) and 
              all other required inputs for the polynomial basis linear model generator (1.b)
      FUNCTION: call 1.b to generate one data point, and update the prior. and
                calculate the paramters of predictive distribution, repeat until
                the posterior probability converges.
      OUTPUT: print the new data point and the current parameters for posterior 
              and predictive distribution.
      HINT: It is not that hard.
'''

def transpose(A):
    AT = np.zeros((np.shape(A)[1],np.shape(A)[0]))
    for i in range(np.shape(A)[1]):
        for j in range(np.shape(A)[0]):
            AT[i,j] = A[j,i]        
    return AT

def BaysianLinRegre(b,n,w,a):    
    index = 1
    # initial
    poster_mean = np.zeros((n,1))
    poster_var = (1./b)*np.eye(n)
    threshold_mean = .001
    threshold_var = .001
    temp_mean = -1*np.ones((n,1))
    temp_var = -1*np.ones((n,n))
    #arr_mean = []
    #arr_var = []
    
    while (sum(abs(poster_mean - temp_mean)) > threshold_mean ) or (sum(sum(abs(poster_var - temp_var))) > threshold_var):  
        x,phi_x,y = basis_lin_model(n,1./a,w)    
        phi_x = np.array(phi_x)#transfer to array type deu to  do a*phi_x        
        xxT = one_rank_matrix(phi_x)        
        invLambda_x = matrix_mul(poster_var,transpose(np.array([phi_x])))
        alpha = dot(phi_x,invLambda_x)#xT{inv(Lambda)}x
        temp_mean = poster_mean
        temp_var = poster_var
        poster_var -= a/(1+ a*alpha)*matrix_mul(matrix_mul(poster_var,xxT),poster_var)
        poster_mean = a/(1+a*alpha)*(invLambda_x)*(y - dot(phi_x,poster_mean)) + poster_mean        
        predictive_mean = dot(phi_x,poster_mean)
        predictive_var = 1./a + alpha
        print("[{}]point: [x:{:.3f} y:{:.3f}] posterior mean:{}".format(index,x,y,transpose(poster_mean)))
        print("variance:\n{}".format(poster_var))        
        print("predictive mean:{} variance:{}".format(predictive_mean,predictive_var))
        index += 1
        #print(sum(abs(poster_mean - temp_mean))) 
        #print(sum(sum(abs(poster_var - temp_var))))
        #print("predict wTy:",x*poster_mean[1,0]+poster_mean[0,0],"\n")

if __name__ =='__main__':
    sequen_estime(3,5)#exercise 2
    BaysianLinRegre(b= 1./3,n= 2,w= [20,26,2],a = 7.)#exercise 3
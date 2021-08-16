# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:15:11 2020

@author: Lenovo
"""

import numpy as np
import copy as cp
import iisignature as sig
#import matplotlib.pyplot as plt

class preprocess:
    def __init__(self):
        with open('F:/signature/pendigits-orig.tes','r')as f:
            self.content_test=f.readlines()
        with open('F:/signature/pendigits-orig.tra','r')as f:
            self.content_train=f.readlines()        
        #test    
        self.data=np.empty([3498,300,2])
        self.index=np.empty([3498])
        self.collection=np.empty([3498,500,500])
        self.collection_25_test=np.empty([3498,25,25])
        self.level=2
        self.sigcollection_test=[]
        #train
        self.data_train=np.empty([7494,300,2])
        self.index_train=np.empty([7494])
        self.sigcollection_train=[]
        self.collection_25_train=np.empty([7494,25,25])
    def process(self):
        #test_process
        n=0
        for i in range(3498):
            while self.content_test[n] !='.PEN_DOWN\n':
                n+=1
            start=cp.deepcopy(n +1)
            num=list(self.content_test[start-3].split()[4])[1]
            while self.content_test[n] !='.PEN_UP\n':
                n+=1
            end = cp.deepcopy(n-1)
            length=end-start+1
            temp=np.zeros([length,2])
            for k in range(length):
                temp[k,:]=self.content_test[start+k].split()
            self.data[i,0:length,:]=temp
            self.index[i]=num
            if n ==161845:
                n=n-4
            while self.content_test[n+2]=='.PEN_DOWN\n':
                start=cp.deepcopy(n+3)
                n=n+3
                while self.content_test[n] !='.PEN_UP\n':
                    n+=1
                end = cp.deepcopy(n-1)
                length1=end-start+1
                temp1=np.zeros([length1,2])
                for k in range(length1):
                    temp1[k,:]=self.content_test[start+k].split()
                self.data[i,length,:]=['0.01','0.01']
                self.data[i,length+1:length+1+length1,:]=temp1
                length=length+1+length1
        #train_process
        n=0
        for i in range(7494):
            while self.content_train[n] !='.PEN_DOWN\n':
                n+=1
            start=cp.deepcopy(n+1)
            num=list(self.content_train[start-3].split()[4])[1]
            while self.content_train[n] !='.PEN_UP\n':
                n+=1
            end = cp.deepcopy(n-1)
            length=end-start+1
            temp=np.zeros([length,2])
            for k in range(length):
                temp[k,:]=self.content_train[start+k].split()
            self.data_train[i,0:length,:]=temp
            self.index_train[i]=num
        
            while self.content_train[n+2]=='.PEN_DOWN\n':
                start=cp.deepcopy(n+3)
                n=n+3
                while self.content_train[n] !='.PEN_UP\n':
                    n+=1
                end = cp.deepcopy(n-1)
                length1=end-start+1
                temp1=np.zeros([length1,2])
                for k in range(length1):
                    temp1[k,:]=self.content_train[start+k].split()
                self.data_train[i,length,:]=['0.01','0.01']
                self.data_train[i,length+1:length+1+length1,:]=temp1
                length=length+1+length1     
                if n==354999:
                    break
    
    def image(self):
        for i in range(3498):
            temp_map=np.zeros([500,500])
            for k in range(300):
                if self.data[i,k,0]==0.1:
                    continue
                if self.data[i,k,0]==0:
                    break
                temp_map[int(self.data[i,k,0]),int(self.data[i,k,1])]=1
            self.collection[i,:,:]=temp_map
            
    def image_25(self):
        #test
        for i in range(3498):
            temp_map=np.zeros([25,25])
            for k in range(300):
                if self.data[i,k,0]==0.01:
                    continue
                if self.data[i,k,0]==0:
                    break
                if self.data[i,k,0]<100 or self.data[i,k,0]>=400:
                    continue
                if self.data[i,k,1]<100 or self.data[i,k,1]>=400:
                    continue
                temp_map[int((self.data[i,k,0]-100)/12),int((self.data[i,k,1]-100)/12)]+=1

            self.collection_25_test[i,:,:]=temp_map   
         #trian
        for i in range(7494):
            temp_map=np.zeros([25,25])
            for k in range(300):
                if self.data_train[i,k,0]==0.01:
                    continue
                if self.data_train[i,k,0]==0:
                    break
                if self.data_train[i,k,0]<100 or self.data_train[i,k,0]>=400:
                    continue
                if self.data_train[i,k,1]<100 or self.data_train[i,k,1]>=400:
                    continue
                temp_map[int((self.data_train[i,k,0]-100)/12),int((self.data_train[i,k,1]-100)/12)]+=1
            self.collection_25_train[i,:,:]=temp_map 
    
    
    def siglayer(self,level=2):
        self.level=level
        depth=2*(2**self.level-1) 
        #test
        self.sigcollection_test=np.zeros([3498,25,25,depth])
        for i in range(3498):
            k=2
            while self.data[i,k+2,0]!=0:
                if self.data[i,k-2:k+3,0].__contains__(0.01):
                    pass
                else:
                    self.sigcollection_test[i,int(self.data[i,k,0]/20),
                                       int(self.data[i,k,1]/20),:]+=sig.sig(
                            self.data[i,k-2:k+3,:],self.level)
                k+=1      
        #train
        self.sigcollection_train=np.zeros([7494,25,25,depth])
        for i in range(7494):
            k=2
            while self.data_train[i,k+2,0]!=0:
                if self.data_train[i,k-2:k+3,0].__contains__(0.01):
                    pass
                else:
                    self.sigcollection_train[i,int(self.data_train[i,k,0]/20),
                                       int(self.data_train[i,k,1]/20),:]+=sig.sig(
                            self.data_train[i,k-2:k+3,:],self.level)
                k+=1
                
if __name__=='__main__':
    a=preprocess()
    a.process()
    a.image_25()
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:21 2014

@author: hp
"""

import string
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats

from Tkinter import *
from tkFileDialog import *
import matplotlib.pyplot



'''
def FILENAME():
    root = Tk() 
    root.withdraw()
    return askopenfilename(parent=root)
'''
def FILE_TO_ARRAY(filename):
    f = open(filename,'r')
    string=f.read()
    f.close()
    return string.split('\n')      
    
def Parse_data(STR):
    Temp=STR.split(',')
    dummy=Temp[0].split('from ')
    A=dummy[1].split(' to ')
    a=int(A[0])
    b=int(A[1])
    
    dummy=Temp[1].split('delay = ')    
    B=dummy[1].split(' at timeslot ')
    c=int(B[0])
    d=int(B[1])
    return a,b,c,d

def MAKE_number_ARRAY(DATA):
    A=[]
    B=[]
    C=[]
    D=[]
    for dat in DATA:
        if dat=='':
            break        
        a,b,c,d=Parse_data(dat)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)        
    return A,B,C,D
   
def column(matrix, i):
    return [row[i] for row in matrix]
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
    
def make_filenamelqf(duration, switch_size, load, iteration, Tth, Qth, ganumber):
    a="d:/2017/rhapsody/PSONSchedulerLQF/component_4/DefaultConfig/sim_result_D"
    a+=str(duration)
    a+="_S"
    a+=str(switch_size)
    a+="_L"
    a+=str(load)
    a+="_I"
    a+=str(iteration)
    a+="_Tth"
    a+=str(Tth)
    a+="_Qth"
    a+=str(Qth)
    a+="_GA"
    a+=str(ganumber)
    a+=".txt"
    return a
    
    
    
def make_filename(duration, switch_size, load, iteration, Tth, Qth, ganumber):
    a="d:/2017/rhapsody/PSONScheduler/component_4/DefaultConfig/sim_result_D"
    a+=str(duration)
    a+="_S"
    a+=str(switch_size)
    a+="_L"
    a+=str(load)
    a+="_I"
    a+=str(iteration)
    a+="_Tth"
    a+=str(Tth)
    a+="_Qth"
    a+=str(Qth)
    a+="_GA"
    a+=str(ganumber)
    a+=".txt"
    return a
    
    
    
    
#filename=FILENAME()
#test_load_list = [52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96, 98, 99]
#test_iter_list = [1,2,3,4,5]
test_load_list = [90]
test_iter_list = [1,2,3,4,5]

'''
test_load_list = [85,90,95]
test_iter_list = [1,2,3]
'''


Delay_threshold = 1280
Que_threshold = 5
Switch_size = 64
Duration = 100000


for i in range(0,len(test_load_list)):
    for j in range(0,len(test_iter_list)):  
  
        print(i,j)
            
           
        filename=make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j],-1, -1, 1)
        DATA = FILE_TO_ARRAY(filename)    
        gaA,gaB,gaC,gaD = MAKE_number_ARRAY(DATA)
    
        filename=make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j],-1, -1, 0)
        DATA = FILE_TO_ARRAY(filename)    
        basicA,basicB,basicC,basicD = MAKE_number_ARRAY(DATA)

        filename=make_filename(Duration, Switch_size, test_load_list[i], test_iter_list[j],Delay_threshold, Que_threshold, 1)
        DATA = FILE_TO_ARRAY(filename)    
        gapaA,gapaB,gapaC,gapaD = MAKE_number_ARRAY(DATA)


        filename=make_filenamelqf(Duration, Switch_size, test_load_list[i], test_iter_list[j],0, 0, 4)
        DATA = FILE_TO_ARRAY(filename)    
        clqfA,clqfB,clqfC,clqfD = MAKE_number_ARRAY(DATA)


        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(4.5, 3.5)
        matplotlib.rc('legend', fontsize=13, labelspacing=0.2, handlelength=2, frameon=False, loc='lower right')    


        #plt.hist(basicC, 1000, normed=1, histtype='step', cumulative=-1,  log=True)
        plt.hist(gaC, 1000, normed=1, histtype='step', cumulative=-1,log=True )
        plt.hist(gapaC, 1000, normed=1, histtype='step', cumulative=-1,  log=True)
        plt.hist(clqfC, 1000, normed=1, histtype='step', cumulative=-1, log=True )
        plt.title('Tail Distribution Load:'+str(test_load_list[i])+' Iteration:'+str(test_iter_list[j]))
        plt.xlabel('Delay (time slots)')
        plt.ylabel('Prob (delay > x)')
        #plt.xlim(3,4000)
        plt.ylim(0.001,1)
        #plt.legend(('Basic','GA', 'GAPA','C-LQF'), loc='upper right')
        plt.legend(('GA', 'GAPA','C-LQF'), loc='upper right')
        #plt.legend(('Without GrantAware', 'With GrantAware'), loc='lower right')
        #plt.legend([basic])
        plt.show()







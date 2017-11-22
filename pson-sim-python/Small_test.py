# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:21 2014

@author: hp
"""

import string
import numpy as np
import matplotlib.pyplot as plt

from Tkinter import *
from tkFileDialog import *




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
    
#filename=FILENAME()
test_load_list = [52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96]
test_iter_list = [1,2,3,4,5]
'''
test_load_list = [85,90,95]
test_iter_list = [1,2,3]
'''

average_delay=[[]]
percentile_delay_90=[[]]
percentile_delay_99=[[]]

temp_average_delay=[]
temp_percentile_delay_90=[]
temp_percentile_delay_99=[]

no_ga_average_delay=[[]]
no_ga_percentile_delay_90=[[]]
no_ga_percentile_delay_99=[[]]

temp_no_ga_average_delay=[]
temp_no_ga_percentile_delay_90=[]
temp_no_ga_percentile_delay_99=[]

ga_average_delay=[[]]
ga_percentile_delay_90=[[]]
ga_percentile_delay_99=[[]]

temp_ga_average_delay=[]
temp_ga_percentile_delay_90=[]
temp_ga_percentile_delay_99=[]





'''
for i in range(0,3):
    for j in range(0,3):    
'''


for i in range(21,23):
    for j in range(0,5):    
        print(i,j)
        #print(test_load_index)
        #print(test_iter_index)
        filename="c:/Users/user/IBM/Rational/Rhapsody/8.1.5/PSONScheduler/component_4/DefaultConfig/sim_result_D40000_S20_L"
        filename+=str(test_load_list[i])
        filename+="_I"
        filename+=str(test_iter_list[j])
        filename+="_Tth-1_Qth-1_GA0.txt"
        DATA = FILE_TO_ARRAY(filename)    
         
        basicA,basicB,basicC,basicD = MAKE_number_ARRAY(DATA)
        temp_average_delay.append(np.average(basicC))
        temp_percentile_delay_90.append(np.percentile(basicC, 90))
        temp_percentile_delay_99.append(np.percentile(basicC, 99))
        
        filename="c:/Users/user/IBM/Rational/Rhapsody/8.1.5/PSONScheduler/component_4/DefaultConfig/sim_result_D40000_S20_L"
        filename+=str(test_load_list[i])
        filename+="_I"
        filename+=str(test_iter_list[j])
        filename+="_Tth-1_Qth-1_GA1.txt"
        DATA = FILE_TO_ARRAY(filename)    
        nogaA,nogaB,nogaC,nogaD = MAKE_number_ARRAY(DATA)
        temp_no_ga_average_delay.append(np.average(nogaC))
        temp_no_ga_percentile_delay_90.append(np.percentile(nogaC, 90))
        temp_no_ga_percentile_delay_99.append(np.percentile(nogaC, 99))
    
        filename="c:/Users/user/IBM/Rational/Rhapsody/8.1.5/PSONScheduler/component_4/DefaultConfig/sim_result_D40000_S20_L"
        filename+=str(test_load_list[i])
        filename+="_I"
        filename+=str(test_iter_list[j])
        filename+="_Tth4000_Qth10_GA1.txt"
        DATA = FILE_TO_ARRAY(filename)    
        gaA,gaB,gaC,gaD = MAKE_number_ARRAY(DATA)
        temp_ga_average_delay.append(np.average(gaC))
        temp_ga_percentile_delay_90.append(np.percentile(gaC, 90))
        temp_ga_percentile_delay_99.append(np.percentile(gaC, 99))
    
        plt.plot(basicC)
        plt.plot(nogaC)
        plt.plot(gaC)
        plt.title('average delay iteration:'+str(test_iter_list[j])+' load:'+str(test_load_list[i]))
        plt.xlabel('time')
        plt.ylabel('Delay (time slots)')
        plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper left')
        plt.show()            


'''    
for i in range(0,1):
    plt.plot(column(average_delay,i))
    plt.plot(column(no_ga_average_delay,i))
    plt.plot(column(ga_average_delay,i))
    plt.title('average delay iteration:'+str(test_iter_list[i]))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper left')
    plt.show()

for i in range(0,1):
    plt.plot(column(percentile_delay_90,i))
    plt.plot(column(no_ga_percentile_delay_90,i))
    plt.plot(column(ga_percentile_delay_90,i))
    plt.title('90% delay iteration:'+str(test_iter_list[i]))
    plt.xlabel('Load')
    plt.ylabel('Delay (time slots)')
    plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper left')
    plt.show()
  
plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()

plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()

plt.plot(column(average_delay,0))
plt.plot(column(no_ga_average_delay,0))
plt.plot(column(ga_average_delay,0))
plt.title('average delay iteration:1')
plt.xlabel('Load')
plt.ylabel('Delay (time slots)')
plt.legend(('Basic','GrantAware', 'Priority GrantAware'), loc='upper right')
plt.show()
'''





